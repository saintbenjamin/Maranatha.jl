# ============================================================================
# src/Quadrature/NewtonCotes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module NewtonCotes

Composite Newton-Cotes backend for rule parsing and exact weight assembly.

# Module description
`NewtonCotes` provides the Newton-Cotes-specific infrastructure used by the
quadrature and error-estimation layers.

Its responsibilities include:

- recognizing and parsing supported Newton-Cotes rule symbols,
- encoding endpoint admissibility constraints through boundary selectors,
- assembling exact composite coefficient vectors using rational arithmetic,
- converting those coefficients into physical 1-dimensional quadrature weights.

The module is also reused by the error-estimation layer when it needs
Newton-Cotes residual models or refinement admissibility checks.

# Notes
- This is an internal module.
- Exact rational assembly is used here to avoid losing structural information
  before the final conversion to floating-point weights.
"""
module NewtonCotes

import ..JobLoggerTools
import ..QuadratureBoundarySpec

"""
    RBig = Rational{BigInt}

Exact rational number type used for composite Newton-Cotes weight assembly.

# Description
`RBig` is defined as:

    Rational{BigInt}

and is used throughout the exact composite Newton-Cotes construction so that
local moments, Vandermonde systems, and the assembled global coefficient vector
remain exact before final conversion to `Float64`.
"""
const RBig = Rational{BigInt}

"""
    _NS_BETA_CACHE :: Dict{Tuple{Int,Symbol,Int,DataType}, Vector}

Cache for composite Newton-Cotes coefficient vectors in the requested scalar type.

# Description
This dictionary stores previously constructed global coefficient vectors ``\\beta``
(after conversion to the active scalar type) so that repeated calls with the same
configuration do not repeat the expensive exact-rational assembly.

# Notes
- Cache key: `(p, boundary, Nsub, real_type)`.
- Stored value: vector of length `Nsub + 1` in the requested scalar type.
- The cache is process-local and not persistent across sessions.
"""
const _NS_BETA_CACHE = Dict{Tuple{Int,Symbol,Int,DataType}, Vector}()

"""
    _local_width(
        p::Int,
        kind::Symbol
    ) -> Int

Return the local block width in units of ``h`` for a Newton-Cotes block.

# Function description
In the exact composite assembly, a local Newton-Cotes block covers a dimensionless
interval ``[0, w]``, where the width depends on the block kind:

- `:closed`  -> ``w = p - 1``
- `:opened`  -> ``w = p``

This width is used when validating the composite tiling constraint and when
constructing exact local moments.

# Arguments
- `p`: Number of nodes in the local block.
- `kind`: Local block type. Must be either `:closed` or `:opened`.

# Returns
- `Int`: Local block width `w` in units of ``h``.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `kind` is unknown.
"""
@inline function _local_width(
    p::Int, 
    kind::Symbol
)
    kind === :closed && return p - 1
    kind === :opened && return p
    JobLoggerTools.error_benji("unknown local kind: $kind")
end

"""
    _local_nodes(
        p::Int,
        kind::Symbol,
        which_open::Symbol
    ) -> Vector{RBig}

Construct the local node positions (dimensionless ``u``) for a Newton-Cotes block.

# Function description
This helper returns the exact rational node locations used in local moment matching.
For closed blocks, the nodes are `0:(p-1)`. For opened blocks, the node placement
depends on the opening direction.

# Arguments
- `p`: Number of local nodes. Must satisfy ``p \\geq 2``.
- `kind`: Block type, either `:closed` or `:opened`.
- `which_open`: Opening direction for opened blocks. Must be `:backward` or `:forward`.

# Returns
- `Vector{RBig}`: Exact local node positions in dimensionless units.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if ``p < 2``.
- Throws via [`JobLoggerTools.error_benji`](@ref) if `kind` is invalid.
- Throws via [`JobLoggerTools.error_benji`](@ref) if `which_open` is invalid for an opened block.
"""
function _local_nodes(
    p::Int, 
    kind::Symbol, 
    which_open::Symbol
)::Vector{RBig}
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2")

    if kind === :closed
        return [RBig(BigInt(i), 1) for i in 0:(p-1)]
    elseif kind === :opened
        if which_open === :backward
            return [RBig(BigInt(i), 1) for i in 1:p]
        elseif which_open === :forward
            return [RBig(BigInt(i), 1) for i in 0:(p-1)]
        else
            JobLoggerTools.error_benji("which_open must be :backward or :forward")
        end
    else
        JobLoggerTools.error_benji("kind must be :closed or :opened")
    end
end

"""
    _exact_moment_0w(
        w::RBig,
        k::Int
    ) -> RBig

Compute the exact monomial moment ``\\displaystyle{\\int\\limits_0^w du \\, u^k}`` as a rational number.

# Function description
This routine evaluates the closed-form moment

```math
\\int\\limits_0^w du \\, u^k = \\frac{w^{k+1}}{k+1}
```

using exact rational arithmetic.

# Arguments
- `w`: Upper limit of the local integration interval, represented as [`RBig`](@ref).
- `k`: Monomial power.

# Returns
- [`RBig`](@ref): Exact value of the monomial moment.

# Errors
- No explicit validation is performed for `k`; unintended inputs may lead to unintended behavior.
"""
@inline function _exact_moment_0w(
    w::RBig, 
    k::Int
)::RBig
    return w^(k+1) / RBig(BigInt(k+1), 1)
end

"""
    _compute_local_alpha(
        p::Int,
        kind::Symbol,
        which_open::Symbol
    ) -> (nodes, α, w_int)

Compute local Newton-Cotes weights ``\\alpha`` on the dimensionless interval ``[0, w]``.

# Function description
This helper builds and solves the exact moment-matching system for a local block.
The local nodes are chosen from [`_local_nodes`](@ref), the interval width is
obtained from [`_local_width`](@ref), and the local weights are determined from
exact monomial moments on ``[0, w]``.

# Arguments
- `p`: Number of nodes in the local block.
- `kind`: Block type, either `:closed` or `:opened`.
- `which_open`: Opening direction used when `kind == :opened`.

# Returns
- `nodes::Vector{RBig}`: Exact local node positions.
- `α::Vector{RBig}`: Exact local Newton-Cotes weights.
- `w_int::Int`: Local block width in units of ``h``.

# Errors
- Propagates validation errors from [`_local_width`](@ref) and [`_local_nodes`](@ref).
- May throw if the exact linear solve fails.
"""
function _compute_local_alpha(
    p::Int, 
    kind::Symbol, 
    which_open::Symbol
)
    w_int = _local_width(p, kind)
    w = RBig(BigInt(w_int), 1)

    nodes = _local_nodes(p, kind, which_open)
    N = length(nodes)  # == p

    A = Matrix{RBig}(undef, N, N)
    q = Vector{RBig}(undef, N)

    for k in 0:(N-1)
        for j in 1:N
            A[k+1, j] = nodes[j]^k
        end
        q[k+1] = _exact_moment_0w(w, k)
    end

    α = A \ q
    return nodes, α, w_int
end

"""
    _check_condition(
        p::Int,
        boundary::Symbol,
        Nsub::Int
    ) -> (m, wL, wR)

Validate and decode the composite tiling constraint for exact Newton-Cotes assembly.

# Function description
The global interval must be tiled by one left boundary block, ``m`` interior closed
blocks, and one right boundary block. Their widths must satisfy the composite
constraint implied by ``p``, `boundary`, and `Nsub`.

# Arguments
- `p`: Local node count. Must satisfy ``p \\geq 2``.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of global subintervals. Must satisfy ``N_\\text{sub} \\geq 1``.

# Returns
- `m::Int`: Number of interior closed blocks.
- `wL::Int`: Width of the left boundary block in units of ``h``.
- `wR::Int`: Width of the right boundary block in units of ``h``.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if ``p < 2``.
- Throws via [`JobLoggerTools.error_benji`](@ref) if ``N_\\text{sub} < 1``.
- Throws via [`JobLoggerTools.error_benji`](@ref) if the boundary tiling constraint is not satisfied.
"""
function _check_condition(
    p::Int, 
    boundary::Symbol, 
    Nsub::Int
)
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2")
    Nsub >= 1 || JobLoggerTools.error_benji("Nsub must be ≥ 1")

    Ltype, Rtype = QuadratureBoundarySpec._decode_boundary(boundary)

    wL = _local_width(p, Ltype)
    wR = _local_width(p, Rtype)
    wC = p - 1

    if Nsub < wL + wR
        JobLoggerTools.error_benji(
            "Invalid Nsub for boundary=$boundary.\n" *
            "Need Nsub ≥ wL + wR = $(wL + wR).\n" *
            "Nearest valid Nsub is $(wL + wR)."
        )
    end

    
    rem = Nsub - wL - wR
    if rem % wC != 0
        m_low  = rem ÷ wC
        m_high = m_low + 1

        N_low  = wL + m_low  * wC + wR
        N_high = wL + m_high * wC + wR

        JobLoggerTools.error_benji(
            "Invalid Nsub for boundary=$boundary.\n" *
            "Require: Nsub = wL + m*(p-1) + wR.\n" *
            "Got: Nsub=$Nsub, wL=$wL, wR=$wR, (p-1)=$wC, remainder=$rem.\n" *
            "Nearest valid Nsub values: $N_low or $N_high."
        )
    end

    m = rem ÷ wC
    return m, wL, wR
end

"""
    _assemble_composite_beta_rational(
        p::Int,
        boundary::Symbol,
        Nsub::Int
    ) -> Vector{RBig}

Assemble the global composite Newton-Cotes coefficient vector `β` in exact rational form.

# Function description
This is the core exact assembly routine. It validates the composite tiling,
constructs the left boundary block, appends interior closed blocks, constructs
the right boundary block, and accumulates all contributions into a single exact
coefficient vector ``\\beta``.

# Arguments
- `p`: Local node count.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of global subintervals.

# Returns
- `Vector{RBig}`: Exact global coefficient vector ``\\beta`` of length `Nsub + 1`.

# Errors
- Propagates validation errors from [`_check_condition`](@ref).
- Throws via [`JobLoggerTools.error_benji`](@ref) if an internal assembly mismatch is detected.
- May emit a warning for large ``p`` because exact rational weights can become extremely large.
"""
function _assemble_composite_beta_rational(
    p::Int, 
    boundary::Symbol, 
    Nsub::Int
)::Vector{RBig}
    m, wL, wR = _check_condition(p, boundary, Nsub)
    Ltype, Rtype = QuadratureBoundarySpec._decode_boundary(boundary)

    if p >= 9
        JobLoggerTools.warn_benji("p=$p is high; NC weights can get enormous (exact rational). May become slow/heavy.")
    end

    β = [RBig(0) for _ in 0:Nsub]  # β[j] stored at β[j+1]
    start = 0

    # Left block
    if Ltype === :closed
        nodesL, αL, w = _compute_local_alpha(p, :closed, :backward)
        @assert w == (p-1)
        for (u, a) in zip(nodesL, αL)
            j = start + Int(u)
            β[j+1] += a
        end
        start += w
    else
        # left-open => backward-open
        nodesL, αL, w = _compute_local_alpha(p, :opened, :backward)
        @assert w == p
        for (u, a) in zip(nodesL, αL)
            j = start + Int(u)
            β[j+1] += a
        end
        start += w
    end

    # Interior closed blocks
    for _ in 1:m
        nodesC, αC, w = _compute_local_alpha(p, :closed, :backward)
        @assert w == (p-1)
        for (u, a) in zip(nodesC, αC)
            j = start + Int(u)
            β[j+1] += a
        end
        start += w
    end

    # Right block
    expected_start = Nsub - wR
    start == expected_start || JobLoggerTools.error_benji("Internal assembly mismatch: start=$start but expected_start=$expected_start")

    if Rtype === :closed
        nodesR, αR, w = _compute_local_alpha(p, :closed, :backward)
        @assert w == (p-1)
        for (u, a) in zip(nodesR, αR)
            j = start + Int(u)
            β[j+1] += a
        end
    else
        # right-open => forward-open
        nodesR, αR, w = _compute_local_alpha(p, :opened, :forward)
        @assert w == p
        for (u, a) in zip(nodesR, αR)
            j = start + Int(u)
            β[j+1] += a
        end
    end

    return β
end

"""
    _is_newton_cotes_rule(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` is a Newton-Cotes rule symbol of the form `:newton_pK`.

# Function description
This helper recognizes the exact composite Newton-Cotes rule symbols used in this module.

# Arguments
- `rule`: Quadrature rule symbol.

# Returns
- `Bool`: `true` if `rule` begins with `"newton_p"`, otherwise `false`.

# Errors
- This function does not throw for invalid rule strings; it only returns `false`.
"""
@inline function _is_newton_cotes_rule(
    rule::Symbol
)::Bool
    startswith(String(rule), "newton_p")
end

"""
    _parse_newton_p(
        rule::Symbol
    ) -> Int

Parse the local node count `p` from a Newton-Cotes rule symbol `:newton_p2`, `:newton_p3`, etc.

# Function description
This helper extracts the integer suffix from a symbol such as `:newton_p3`
and validates that the parsed node count is usable for this module.

# Arguments
- `rule`: Quadrature rule symbol expected to start with `"newton_p"`.

# Returns
- `Int`: Parsed local node count `p`.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `rule` does not begin with `"newton_p"`.
- Throws via [`JobLoggerTools.error_benji`](@ref) if the parsed `p` is smaller than `2`.
- May throw if the numeric suffix cannot be parsed as an integer.
"""
function _parse_newton_p(
    rule::Symbol
)::Int
    s = String(rule)
    _is_newton_cotes_rule(rule) || JobLoggerTools.error_benji("rule must start with :newton_p (got $rule)")
    p = parse(Int, s[9:end])  # after "newton_p"
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2 (got p=$p in rule=$rule)")
    return p
end

"""
    _get_beta(
        T,
        p::Int,
        boundary::Symbol,
        Nsub::Int
    )

Get the global composite coefficient vector ``\\beta`` in the requested scalar type, with caching.

# Function description
This routine wraps the exact rational assembly in a typed interface.
It first checks [`_NS_BETA_CACHE`](@ref), assembles the exact rational coefficient
vector if necessary, converts it to the requested scalar type `T`, stores it in
the cache, and returns it.

# Arguments
- `T`: Target scalar type for the returned coefficient vector.
- `p`: Local node count.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of global subintervals.

# Returns
- Coefficient vector ``\\beta`` in scalar type `T`.

# Errors
- Propagates validation and assembly errors from [`_assemble_composite_beta_rational`](@ref).
- May emit warnings inherited from the exact assembly path.
"""
function _get_beta(
    T,
    p::Int,
    boundary::Symbol,
    Nsub::Int
)
    key = (p, boundary, Nsub, T)
    cached = get(_NS_BETA_CACHE, key, nothing)
    cached !== nothing && return cached

    βR = _assemble_composite_beta_rational(p, boundary, Nsub)
    βT = Vector{T}(undef, length(βR))
    @inbounds for i in eachindex(βR)
        βT[i] = convert(T, βR[i])
    end

    _NS_BETA_CACHE[key] = βT
    return βT
end

"""
    _nearest_valid_Nsub(
        p::Int,
        boundary::Symbol,
        Nsub::Int
    ) -> Int

Return the nearest boundary-compatible composite subdivision count for a
Newton-Cotes rule.

# Function description
Newton-Cotes composite assembly does not allow arbitrary subdivision counts.
For a given local node count `p` and boundary pattern `boundary`, the global
subdivision count must satisfy the tiling constraint

```math
N_{\\mathrm{sub}} = w_L + m (p-1) + w_R,
```

where `w_L` and `w_R` are the left and right boundary block widths and `m` is a
nonnegative integer counting interior closed blocks.

This helper maps an input candidate `Nsub` to the nearest valid subdivision
count compatible with that structure.

# Arguments

* `p::Int`:
  Local node count of the Newton-Cotes rule.
* `boundary::Symbol`:
  Boundary pattern symbol.
* `Nsub::Int`:
  Candidate subdivision count to be adjusted.

# Returns

* `Int`:
  Nearest valid subdivision count compatible with `(p, boundary)`.

# Errors

* Propagates boundary-validation errors from
  `QuadratureBoundarySpec._decode_boundary`.
* Propagates errors from [`_local_width`](@ref) if the decoded local block type
  is invalid.

# Notes

* If `Nsub <= w_L + w_R`, the smallest valid subdivision count `w_L + w_R` is
  returned.
* Otherwise, the function rounds to the nearest admissible value in the
  arithmetic progression `w_L + w_R + m(p-1)`.
* Unlike [`_next_valid_Nsub`](@ref), this helper may return a valid value
  strictly smaller than the input `Nsub`.
"""
function _nearest_valid_Nsub(
    p::Int,
    boundary::Symbol,
    Nsub::Int
)
    Ltype, Rtype = QuadratureBoundarySpec._decode_boundary(boundary)

    wL = _local_width(p, Ltype)
    wR = _local_width(p, Rtype)
    wC = p - 1

    base = wL + wR

    if Nsub <= base
        return base
    end

    offset = Nsub - base
    m = round(Int, offset / wC)

    return base + m * wC
end

"""
    _next_valid_Nsub(
        p::Int,
        boundary::Symbol,
        Nsub::Int
    ) -> Int

Return the smallest boundary-compatible composite subdivision count greater than
or equal to `Nsub`.

# Function description
Newton-Cotes composite assembly requires the global subdivision count to satisfy

```math
N_{\\mathrm{sub}} = w_L + m (p-1) + w_R,
```

with the same boundary-dependent block widths used by
[`_nearest_valid_Nsub`](@ref).

This helper is the monotone counterpart of [`_nearest_valid_Nsub`](@ref): it
never rounds downward. It is intended for callers that treat `Nsub` as a lower
bound, such as refinement and residual-model setup.

# Arguments

* `p::Int`:
  Local node count of the Newton-Cotes rule.
* `boundary::Symbol`:
  Boundary pattern symbol.
* `Nsub::Int`:
  Requested lower bound for the subdivision count.

# Returns

* `Int`:
  The smallest valid subdivision count compatible with `(p, boundary)` such
  that the returned value is `>= Nsub`.

# Errors

* Propagates boundary-validation errors from
  `QuadratureBoundarySpec._decode_boundary`.
* Propagates errors from [`_local_width`](@ref) if the decoded local block type
  is invalid.

# Notes

* If `Nsub <= w_L + w_R`, the function returns the smallest admissible count
  `w_L + w_R`.
* Otherwise, the function rounds upward within the admissible arithmetic
  progression `w_L + w_R + m(p-1)`.
"""
function _next_valid_Nsub(
    p::Int,
    boundary::Symbol,
    Nsub::Int
)
    Ltype, Rtype = QuadratureBoundarySpec._decode_boundary(boundary)

    wL = _local_width(p, Ltype)
    wR = _local_width(p, Rtype)
    wC = p - 1

    base = wL + wR

    if Nsub <= base
        return base
    end

    offset = Nsub - base
    m = cld(offset, wC)

    return base + m * wC
end

end  # module NewtonCotes
