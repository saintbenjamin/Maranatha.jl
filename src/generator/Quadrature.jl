# ============================================================================
# src/rules/Integrate.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Integrate

using LinearAlgebra
using ..JobLoggerTools

export integrate, quadrature_1d_nodes_weights

include("Integrate/integrate_1d.jl")
include("Integrate/integrate_2d.jl")
include("Integrate/integrate_3d.jl")
include("Integrate/integrate_4d.jl")
include("Integrate/integrate_nd.jl")

"""
    integrate(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary
    ) -> Float64

Evaluate a tensor-product Newton-Cotes quadrature on the hypercube ``[a,b]^{\\texttt{dim}}``.

# Function description
This function serves as the unified integration dispatcher within the `Maranatha.jl` pipeline.

1) It builds the **1D nodes and weights** for the selected Newton-Cotes `rule`
   on ``[a,b]`` with resolution `N`.
2) It evaluates the **tensor-product quadrature** in `dim` dimensions by
   enumerating the multi-index over the 1D nodes and accumulating the weighted
   sum of ``\\texttt{integrand}(x_1,\\,\\ldots,\\,x_{\\texttt{dim}})``.

The same bounds ``[a,b]`` are applied along every axis, i.e. the integration domain
is ``[a,b]^{\\texttt{dim}}``.

# Arguments
- `integrand`: A callable that accepts exactly `dim` positional arguments
  (function, closure, or callable struct).
- `a`, `b`: Lower/upper bounds applied to every axis.
- `N`: Number of subintervals per axis (rule-specific constraints apply).
- `dim`: Dimensionality (must satisfy `dim ≥ 1`).
- `rule`: Quadrature rule symbol (e.g. `:simpson13_close`, `:simpson38_open`, `:bode_close`, ...).

# Returns
- `Float64`: Estimated integral value.

# Errors
- Throws an error if `dim < 1`.
- Throws an error if `rule` is unknown or if `N` violates rule-specific constraints.
- Any error thrown by `integrand` during evaluation is propagated.
"""
function integrate(
    integrand, 
    a, 
    b, 
    N, 
    dim, 
    rule,
    boundary
)
    if dim == 1
        return integrate_1d(integrand, a, b, N, rule, boundary)
    elseif dim == 2
        return integrate_2d(integrand, a, b, N, rule, boundary)
    elseif dim == 3
        return integrate_3d(integrand, a, b, N, rule, boundary)
    elseif dim == 4
        return integrate_4d(integrand, a, b, N, rule, boundary)
    else
        return integrate_nd(integrand, a, b, N, rule, boundary; dim=dim)
    end
end

# ============================================================
# Composite Newton–Cotes via exact rational assembly
# rule  : :ns_p3, :ns_p4, :ns_p5, ...
# boundary : :LCRC | :LORC | :LCRO | :LORO
#
# Produces Float64 nodes/weights:
#   xs[j] = a + j*(b-a)/N,  j=0..N
#   ws[j] = β[j] * h,       h=(b-a)/N
# ============================================================

"""
    RBig = Rational{BigInt}

Exact rational number type used for composite Newton-Cotes weight assembly.

# Description
`RBig` is defined as:

    Rational{BigInt}

and is used throughout the exact composite Newton-Cotes construction
to guarantee that:

- All local moment integrals are computed exactly.
- All Vandermonde systems for local weights ``\\alpha`` are solved in exact arithmetic.
- The assembled global coefficient vector ``\\beta`` is mathematically exact
  before conversion to `Float64`.

This prevents any floating-point contamination during symbolic-like
weight construction.

# Design rationale
Composite Newton-Cotes weights can involve large rational coefficients.
Using `Rational{BigInt}` ensures:

- No loss of precision during assembly.
- Exact cancellation between overlapping local blocks.
- Deterministic reproducibility independent of floating-point rounding.

Conversion to `Float64` happens only in [`_get_beta_float`](@ref).
"""
const RBig = Rational{BigInt}

"""
    _NS_BETA_CACHE :: Dict{Tuple{Int,Symbol,Int}, Vector{Float64}}

Cache for `Float64` composite Newton-Cotes coefficient vectors.

# Description
This dictionary stores previously constructed global coefficient vectors ``\\beta``
(after conversion to `Float64`) to avoid repeated expensive exact-rational
assembly.

The cache key is:

    (p, boundary, Nsub)

where:
- `p`        : local Newton-Cotes node count (e.g. 3, 4, 5, ...)
- `boundary` : boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`)
- `Nsub`     : number of global subintervals

The stored value is:

    Vector{Float64}  # length Nsub + 1

representing the global coefficient vector `β` such that:
```math
\\texttt{ws}_j = \\beta_j \\, h
```
with ``\\displaystyle{h = \\frac{b-a}{N_{\\text{sub}}}}``.

# Purpose
Exact rational assembly via [`_assemble_composite_beta_rational`](@ref)
can be computationally heavy for large ``p`` or repeated calls.
This cache ensures that identical quadrature configurations
reuse previously computed weights.

# Notes
- The cache is process-local (not persistent across sessions).
- Memory usage grows with distinct `(p, boundary, Nsub)` triples.
- Safe for repeated deterministic use since values are immutable `Vector{Float64}`.
"""
const _NS_BETA_CACHE = Dict{Tuple{Int,Symbol,Int}, Vector{Float64}}()

"""
    _decode_boundary(
        boundary::Symbol
    ) -> Tuple{Symbol,Symbol}

Decode a composite boundary pattern symbol into the left/right local rule kinds.

# Function description
This helper maps the global boundary pattern used by the exact-rational
composite Newton-Cotes assembly into the *local* endpoint kinds:

- `:closed` means the local block includes the endpoint node.
- `:opened` means the local block is shifted (open-type block).

Supported boundary patterns are:
- `:LCRC` (Left Closed, Right Closed)
- `:LORC` (Left Opened, Right Closed)
- `:LCRO` (Left Closed, Right Opened)
- `:LORO` (Left Opened, Right Opened)

# Arguments
- `boundary`: Boundary pattern symbol.

# Returns
- `(Ltype, Rtype)`: A tuple of symbols, each either `:closed` or `:opened`.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `boundary` is not one of the supported symbols.
"""
@inline function _decode_boundary(
    boundary::Symbol
)
    if boundary === :LCRC
        return (:closed, :closed)
    elseif boundary === :LORC
        return (:opened, :closed)
    elseif boundary === :LCRO
        return (:closed, :opened)
    elseif boundary === :LORO
        return (:opened, :opened)
    else
        JobLoggerTools.error_benji("boundary must be one of: :LCRC | :LORC | :LCRO | :LORO (got $boundary)")
    end
end

"""
    _local_width(
        p::Int, 
        kind::Symbol
    ) -> Int

Return the local block width in units of ``h`` for a Newton-Cotes block of order ``p``.

# Function description
In the exact-rational composite assembly, each local Newton-Cotes block covers
a domain ``[0, w]`` in dimensionless units, where ``w`` depends on whether the block
is closed or opened:

- closed block: ``w = p - 1``
- opened block: ``w = p``

This width is used to enforce the composite tiling constraint and to build the
exact moment vector on ``[0, w]``.

# Arguments
- `p`: Number of nodes in the local Newton-Cotes block (must satisfy ``p \\ge 2``).
- `kind`: Either `:closed` or `:opened`.

# Returns
- `Int`: The local width `w` in units of `h`.

# Errors
- Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `kind` is unknown.
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

Construct the local node positions (dimensionless `u`) for a Newton-Cotes block.

# Function description
This helper produces the exact (rational) node locations used to solve for the
local quadrature weights ``\\alpha`` via moment matching.

- For `kind == :closed`, nodes are `u = 0:(p-1)` (length `p`).
- For `kind == :opened`, nodes depend on the direction:
  - `which_open == :backward`: `u = 1:p` (left-opened style)
  - `which_open == :forward` : `u = 0:(p-1)` (right-opened style)

All nodes are returned as `RBig = Rational{BigInt}` to support exact assembly.

# Arguments
- `p`: Number of nodes (must satisfy ``p \\ge 2``).
- `kind`: Either `:closed` or `:opened`.
- `which_open`: For opened blocks, either `:backward` or `:forward`.

# Returns
- `Vector{RBig}`: Local nodes in dimensionless units.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if ``p < 2``, `kind` is invalid,
  or `which_open` is not one of `:backward` / `:forward` when needed.
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

Compute the exact monomial moment ``\\displaystyle{\\int\\limits_0^w du \\; u^k}`` as a rational number.

# Function description
This routine returns the closed-form value
```math
\\displaystyle{\\int\\limits_0^w du \\; u^k = \\frac{w^{k+1}}{k+1}}
```
using exact rational arithmetic ([`RBig`](@ref)) to avoid any rounding during the
moment-matching solve for local Newton-Cotes weights.

# Arguments
- `w`: Upper limit of the local integration interval (dimensionless), as [`RBig`](@ref).
- `k`: Monomial power (assumed ``k \\ge 0`` in intended usage).

# Returns
- [`RBig`](@ref): The exact moment value.

# Errors
- No explicit checks are performed; invalid `k` may lead to unintended behavior.
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
This helper constructs the moment-matching linear system for a local block:

- Choose local nodes ``u_j`` depending on `kind` / `which_open`.
- Set ``w`` (in units of ``h``) via [`_local_width`](@ref)`(p, kind)`.
- Solve the Vandermonde-like system:
```math
\\sum_j \\alpha_j \\, u_j^k = \\int\\limits_0^w du \\; u^k \\, \\quad \\text{for} \\quad k = 0 , \\ldots , p-1
```

All computations are performed in exact rational arithmetic ([`RBig`](@ref)) so that the
assembled composite weights are exact before conversion to `Float64`.

# Arguments
- `p`: Number of nodes in the local block (must satisfy ``p \\ge 2``).
- `kind`: Either `:closed` or `:opened`.
- `which_open`: For opened blocks, either `:backward` or `:forward`.

# Returns
- `nodes::Vector{RBig}`: Local node positions in dimensionless units.
- `α::Vector{RBig}`: Exact local weights satisfying moment matching on ``[0, w]``.
- `w_int::Int`: Local interval width `w` in units of `h`.

# Errors
- Throws (via downstream helpers) if arguments are invalid or if the solve fails.
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
The composite construction must tile the global interval ``[0, N_\\text{sub}]`` (in units of ``h``)
using:

- one left boundary block of width ``w_L``,
- ``m`` interior closed blocks each of width ``p-1``,
- one right boundary block of width ``w_R``.

The required constraint is:
```math
N_\\text{sub} = w_L + m \\, (p - 1) + w_R
```
where ``w_L`` / ``w_R`` depend on the boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

This function checks:
1) ``N_\\text{sub} \\ge w_L + w_R``, and
2) ``(N_\\text{sub} - w_L - w_R)`` divisible by ``(p-1)``.

If invalid, it throws with a message that includes nearby valid ``N_\\text{sub}`` values.

# Arguments
- `p`: Local node count (must satisfy ``p \\ge 2``).
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of subintervals for the global composite rule (must satisfy ``N_\\text{sub} \\ge 1``).

# Returns
- `m::Int`: Number of interior closed blocks.
- `wL::Int`: Left block width in units of ``h``.
- `wR::Int`: Right block width in units of ``h``.

# Errors
- Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if constraints are violated.
"""
function _check_condition(
    p::Int, 
    boundary::Symbol, 
    Nsub::Int
)
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2")
    Nsub >= 1 || JobLoggerTools.error_benji("Nsub must be ≥ 1")

    Ltype, Rtype = _decode_boundary(boundary)

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

Assemble the global composite Newton-Cotes coefficient vector ``\\beta`` in exact rational form.

# Function description
This is the core exact assembly routine. It builds a global coefficient vector ``\\beta``
(length ``N_\\text{sub}+1``) such that the composite quadrature weights are:
```math
\\texttt{ws}_j = \\beta_j \\, h \\, \\quad \\text{with} \\quad h = \\frac{b-a}{N_\\text{sub}}
```

The algorithm:
1) Validates the tiling constraint via [`_check_condition`](@ref).
2) Builds the left boundary block (closed or opened as requested).
3) Adds ``m`` interior closed blocks (each width ``p-1``).
4) Builds the right boundary block (closed or opened as requested).
5) Accumulates all contributions into a single global ``\\beta``.

All computations remain in `RBig = Rational{BigInt}` to preserve exactness.

# Arguments
- `p`: Local node count (NC *order* in this implementation; must satisfy ``p \\ge 2``).
- `boundary`: Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
- `Nsub`: Number of global subintervals (must satisfy the composite constraint).

# Returns
- `Vector{RBig}`: Exact global coefficient vector ``\\beta`` of length ``N_\\text{sub}+1``.

# Errors
- Throws (via helper checks) if the composite constraint fails or if internal assembly mismatches occur.

# Performance notes
- For large `p` the exact rational weights can become extremely large; this may be slow
  and memory-heavy even before conversion to `Float64`.
"""
function _assemble_composite_beta_rational(
    p::Int, 
    boundary::Symbol, 
    Nsub::Int
)::Vector{RBig}
    m, wL, wR = _check_condition(p, boundary, Nsub)
    Ltype, Rtype = _decode_boundary(boundary)

    if p >= 9
        @warn "p=$p is high; NC weights can get enormous (exact rational). May become slow/heavy."
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
    _is_ns_rule(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` is a composite exact-assembly Newton-Cotes rule symbol of the form `:ns_pK`.

# Function description
This helper recognizes the new composite exact-rational rules introduced in this module.
A rule is considered an *NS rule* if its symbol string begins with `"ns_p"` (e.g. `:ns_p3`, `:ns_p5`).

# Arguments
- `rule`: Quadrature rule symbol.

# Returns
- `Bool`: `true` if `rule` starts with `"ns_p"`, else `false`.
"""
@inline function _is_ns_rule(
    rule::Symbol
)::Bool
    startswith(String(rule), "ns_p")
end

"""
    _parse_ns_p(
        rule::Symbol
    ) -> Int

Parse the local node count `p` from an NS rule symbol `:ns_pK`.

# Function description
For composite exact-assembly Newton-Cotes rules, the rule symbol encodes the
local node count `p` as:

    :ns_p3, :ns_p4, :ns_p5, ...

This function extracts and validates `p` from the symbol.

# Arguments
- `rule`: Quadrature rule symbol, expected to start with `"ns_p"`.

# Returns
- `Int`: Parsed node count `p` (guaranteed ``p \\ge 2`` if successful).

# Errors
- Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if the symbol does not start with `"ns_p"`
  or if the parsed `p` is invalid.
"""
function _parse_ns_p(
    rule::Symbol
)::Int
    s = String(rule)
    _is_ns_rule(rule) || JobLoggerTools.error_benji("rule must start with :ns_p (got $rule)")
    p = parse(Int, s[5:end])  # after "ns_p"
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2 (got p=$p in rule=$rule)")
    return p
end

"""
    _get_beta_float(
        p::Int, 
        boundary::Symbol, 
        Nsub::Int
    ) -> Vector{Float64}

Get the global composite coefficient vector ``\\beta`` in `Float64`, with optional caching.

# Function description
This routine is the `Float64`-facing wrapper around the exact rational assembly:

1) Check the cache [`_NS_BETA_CACHE`](@ref) using key `(p, boundary, Nsub)`.
2) If missing, build ``\\beta`` exactly via [`_assemble_composite_beta_rational`](@ref).
3) Convert each [`RBig`](@ref) entry to `Float64`.
4) Store the result in the cache and return it.

The returned vector ``\\beta`` has length ``N_\\text{sub}+1`` and is intended to be scaled by ``h`` to produce
quadrature weights `ws`.

# Arguments
- `p`: Local node count (must satisfy ``p \\ge 2``).
- `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
- `Nsub`: Number of subintervals (must satisfy the composite constraint for the given boundary).

# Returns
- `Vector{Float64}`: The coefficient vector ``\\beta`` (length ``N_\\text{sub}+1``) in `Float64`.

# Errors
- Propagates any error from [`_assemble_composite_beta_rational`](@ref) and its validators.

# Performance notes
- The cache is recommended when repeatedly calling the same `(p, boundary, Nsub)`.
"""
function _get_beta_float(
    p::Int, 
    boundary::Symbol, 
    Nsub::Int
)::Vector{Float64}
    key = (p, boundary, Nsub)
    cached = get(_NS_BETA_CACHE, key, nothing)
    cached !== nothing && return cached

    βR = _assemble_composite_beta_rational(p, boundary, Nsub)
    βF = Vector{Float64}(undef, length(βR))
    @inbounds for i in eachindex(βR)
        βF[i] = Float64(βR[i])
    end

    _NS_BETA_CACHE[key] = βF
    return βF
end

"""
    quadrature_1d_nodes_weights(
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol, 
        boundary::Symbol
    ) -> (xs, ws)

Construct ``1``-dimensional quadrature nodes and weights on ``[a, b]`` for composite Newton-Cotes rules.

# Function description
This function is the public ``1``-dimensional node/weight generator used by the integration dispatchers.

It supports:

## Composite exact-assembly rules `:ns_pK`
If `rule` is recognized as an NS rule, this routine:
1) Parses `p` from `rule`,
2) Builds (or fetches) the coefficient vector `β` for `(p, boundary, N)`,
3) Forms nodes ``\\texttt{xs}_j = a + j \\, h`` for ``j = 0 , \\ldots , N``,
4) Forms weights ``\\texttt{ws}_j = \\beta_j \\, h``, where ``\\displaystyle{h = \\frac{b-a}{N}}``.

The return types are `Vector{Float64}` for both nodes and weights.

# Arguments
- `a`, `b`: Lower/upper bounds of the 1D interval.
- `N`: Number of subintervals (must satisfy ``N \\ge 1`` and composite constraints for the selected boundary).
- `rule`: Rule symbol. Supported:
  - New rules: `:ns_p3`, `:ns_p4`, `:ns_p5`, ...
- `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
  Required for NS rules.

# Returns
- `xs::Vector{Float64}`: Nodes of length ``N+1``.
- `ws::Vector{Float64}`: Weights of length ``N+1``.

# Errors
- Throws `ArgumentError` if ``N < 1``.
- Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid,
  if the composite constraint fails, or if `rule` is unsupported.

# Notes
- This function currently errors on non-NS rules unless you extend the fallback branch
  with your pre-existing implementation.
"""
function quadrature_1d_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol
)::Tuple{Vector{Float64}, Vector{Float64}}

    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    # boundary validation early (even if legacy overrides it)
    _decode_boundary(boundary)

    # --- composite NS branch ---
    if _is_ns_rule(rule)
        p = _parse_ns_p(rule)
        β = _get_beta_float(p, boundary, N)

        aa = Float64(a)
        bb = Float64(b)
        h = (bb - aa) / Float64(N)

        xs = collect(range(aa, bb; length=N+1))
        ws = Vector{Float64}(undef, N+1)
        @inbounds for j in 0:N
            ws[j+1] = β[j+1] * h
        end

        return xs, ws
    end

    # ------------------------------------------------------------
    # FALLBACK: existing non-NS rules (keep your current code here)
    # ------------------------------------------------------------
    JobLoggerTools.error_benji("Unsupported rule=$rule (and not recognized as :ns_pK).")
end

end  # module Integrate