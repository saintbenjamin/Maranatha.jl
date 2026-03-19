# ============================================================================
# src/Quadrature/Gauss.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Gauss

import ..LinearAlgebra

import ..JobLoggerTools

"""
    _legendre_Pn_Pn1(
        n::Int,
        x::Float64
    ) -> (P_n, P_{n-1})

Evaluate the standard Legendre polynomial ``P_n(x)`` and the previous-order value
``P_{n-1}(x)`` at a scalar point ``x``.

# Function description
This low-level helper evaluates Legendre polynomials using the standard three-term
recurrence and returns the pair ``(P_n(x), P_{n-1}(x))``. It is mainly used inside
Newton iterations for Gauss-Radau and Gauss-Lobatto root solves.

# Arguments
- `n`: Polynomial order (must satisfy ``n \\geq 0``).
- `x`: Evaluation point.

# Returns
- `Tuple{Float64,Float64}`: The pair ``(P_n(x), P_{n-1}(x))``.

# Errors
- Throws `error("n must be ≥ 0")` if `n < 0`.
"""
@inline function _legendre_Pn_Pn1(
    n::Int, 
    x::Float64
)::Tuple{Float64,Float64}
    n >= 0 || error("n must be ≥ 0")
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end

    Pnm1 = 1.0    # P0
    Pn   = x      # P1
    for k in 2:n
        Pkp = ((2k - 1) * x * Pn - (k - 1) * Pnm1) / k
        Pnm1 = Pn
        Pn   = Pkp
    end
    return Pn, Pnm1
end

"""
    _legendre_Pn_deriv(
        n::Int,
        x::Float64
    ) -> (P_n, P_n′)

Evaluate the standard Legendre polynomial ``P_n(x)`` and its derivative ``P_n^\\prime(x)``.

# Function description
This helper first evaluates ``P_n(x)`` and ``P_{n-1}(x)`` via
[`_legendre_Pn_Pn1`](@ref), then computes the derivative using the standard identity
for Legendre polynomials. It is used in Newton iterations for Radau and Lobatto
root finding.

# Arguments
- `n`: Polynomial order (must satisfy ``n \\geq 0``).
- `x`: Evaluation point.

# Returns
- `Tuple{Float64,Float64}`: The pair ``(P_n(x), P_n^\\prime(x))``.

# Errors
- Propagates any error from [`_legendre_Pn_Pn1`](@ref), including invalid `n`.

# Notes
- For `n == 0`, this returns `(1.0, 0.0)`.
- Near ``x = \\pm 1``, the derivative formula is numerically delicate because it
  contains a factor proportional to ``(x^2 - 1)^{-1}``.
"""
@inline function _legendre_Pn_deriv(
    n::Int, 
    x::Float64
)::Tuple{Float64,Float64}
    # returns (P_n(x), P_n'(x))
    (Pn, Pn1) = _legendre_Pn_Pn1(n, x)
    if n == 0
        return Pn, 0.0
    end
    # Pn'(x) = n/(x^2 - 1) * (x*Pn - P_{n-1})
    den = x*x - 1.0
    # avoid blow-up extremely near ±1 in Newton; caller clamps away from endpoints
    dPn = n * (x * Pn - Pn1) / den
    return Pn, dPn
end

"""
    _clamp_open(
        x::Float64
    ) -> Float64

Clamp a Newton iterate away from the singular endpoints ``x = \\pm 1``.

# Function description
Several derivative identities used in Gauss-family root finding become singular at
``x = \\pm 1``. This helper clamps a floating-point iterate into a safe open interval
strictly inside ``[-1,1]`` for numerical stability during Newton updates.

# Arguments
- `x`: Newton iterate.

# Returns
- `Float64`: Clamped value lying inside a small open neighborhood of ``(-1,1)``.

# Errors
- This function does not throw explicitly.
"""
@inline function _clamp_open(
    x::Float64
)::Float64
    # keep Newton away from ±1 where derivative formula divides by (x^2-1)
    # (this is purely numerical hygiene)
    tiny = 64.0 * eps(Float64)
    x <= -1.0 + tiny && return -1.0 + tiny
    x >=  1.0 - tiny && return  1.0 - tiny
    return x
end

"""
    gauss_legendre_nodes_weights_float(
        n::Int
    ) -> (nodes, weights)

Compute the ``n``-point Gauss-Legendre rule on ``[-1,1]`` in `Float64`.

# Function description
This routine constructs the single-interval Gauss-Legendre quadrature rule using the
Golub-Welsch algorithm, i.e. the eigen-decomposition of the symmetric tridiagonal
Jacobi matrix associated with the Legendre weight ``w(x)=1``.

# Arguments
- `n`: Number of quadrature points (must satisfy ``n \\geq 1``).

# Returns
- `nodes::Vector{Float64}`: Length-`n` node vector on ``[-1,1]``, sorted ascending.
- `weights::Vector{Float64}`: Length-`n` weight vector aligned with `nodes`.

# Errors
- Throws `error("n must be ≥ 1")` if ``n < 1``.

# Notes
- This function returns the single-interval rule only. Composite repetition is handled
  by higher-level helpers such as [`_composite_gauss_nodes_weights`](@ref).
"""
function gauss_legendre_nodes_weights_float(
    n::Int
)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 1 || error("n must be ≥ 1")

    a = zeros(Float64, n)
    b = Vector{Float64}(undef, n-1)
    @inbounds for k in 1:(n-1)
        b[k] = k / sqrt(4.0*k*k - 1.0)
    end

    J = LinearAlgebra.SymTridiagonal(a, b)
    E = LinearAlgebra.eigen(J)  # sorted ascending
    t = Vector{Float64}(E.values)
    V = E.vectors
    w = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        v1 = V[1, i]
        w[i] = 2.0 * (v1 * v1)
    end
    return t, w
end

"""
    gauss_radau_left_nodes_weights_float(
        n::Int
    ) -> (nodes, weights)

Compute the ``n``-point Gauss-Radau rule on ``[-1,1]`` that includes the left endpoint
``x=-1``.

# Function description
This routine constructs the left-Radau rule for Legendre weight ``w(x)=1``. The left
endpoint is fixed, while the remaining ``n-1`` interior nodes are obtained as roots of
``P_n(x) + P_{n-1}(x)`` via Newton iteration. Interior weights are then computed from
standard closed-form Radau formulas.

# Arguments
- `n`: Number of quadrature points (must satisfy ``n \\geq 2``).

# Returns
- `nodes::Vector{Float64}`: Length-`n` node vector, sorted ascending, including `-1.0`.
- `weights::Vector{Float64}`: Length-`n` weight vector aligned with `nodes`.

# Errors
- Throws `error("Radau needs n ≥ 2 ...")` if ``n < 2``.

# Notes
- Newton iterates are clamped by [`_clamp_open`](@ref) for numerical safety.
"""
function gauss_radau_left_nodes_weights_float(
    n::Int
)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Radau needs n ≥ 2 (got n=$n)")

    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[1] = -1.0
    w[1] = 2.0 / (Float64(n) * Float64(n))

    seeds, _ = gauss_legendre_nodes_weights_float(n-1)

    @inbounds for i in 1:(n-1)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pn, dPn)     = _legendre_Pn_deriv(n, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            F  = Pn + Pnm1
            dF = dPn + dPnm1

            dx = F / dF
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end
        t[i+1] = x

        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i+1] = (1.0 + x) / ((Float64(n)^2) * (Pnm1*Pnm1))
    end

    p = sortperm(t)
    return t[p], w[p]
end

"""
    gauss_radau_right_nodes_weights_float(
        n::Int
    ) -> (nodes, weights)

Compute the ``n``-point Gauss-Radau rule on ``[-1,1]`` that includes the right endpoint
``x=+1``.

# Function description
This routine constructs the right-Radau rule for Legendre weight ``w(x)=1``. The right
endpoint is fixed, while the remaining ``n-1`` interior nodes are obtained as roots of
``P_n(x) - P_{n-1}(x)`` via Newton iteration. Interior weights are then computed from
standard closed-form Radau formulas.

# Arguments
- `n`: Number of quadrature points (must satisfy ``n \\geq 2``).

# Returns
- `nodes::Vector{Float64}`: Length-`n` node vector, sorted ascending, including `+1.0`.
- `weights::Vector{Float64}`: Length-`n` weight vector aligned with `nodes`.

# Errors
- Throws `error("Radau needs n ≥ 2 ...")` if ``n < 2``.

# Notes
- Newton iterates are clamped by [`_clamp_open`](@ref) for numerical safety.
"""
function gauss_radau_right_nodes_weights_float(
    n::Int
)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Radau needs n ≥ 2 (got n=$n)")

    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[end] = 1.0
    w[end] = 2.0 / (Float64(n) * Float64(n))

    seeds, _ = gauss_legendre_nodes_weights_float(n-1)

    @inbounds for i in 1:(n-1)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pn, dPn)     = _legendre_Pn_deriv(n, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            F  = Pn - Pnm1
            dF = dPn - dPnm1

            dx = F / dF
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end
        t[i] = x

        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i] = (1.0 - x) / ((Float64(n)^2) * (Pnm1*Pnm1))
    end

    p = sortperm(t)
    return t[p], w[p]
end

"""
    gauss_lobatto_nodes_weights_float(
        n::Int
    ) -> (nodes, weights)

Compute the ``n``-point Gauss-Lobatto rule on ``[-1,1]`` that includes both endpoints.

# Function description
This routine constructs the Lobatto rule for Legendre weight ``w(x)=1``. The endpoints
``-1`` and ``+1`` are fixed, and the interior nodes are obtained from the zeros of the
Lobatto interior equation solved by Newton iteration. The corresponding weights are then
assembled from the standard closed-form formulas.

# Arguments
- `n`: Number of quadrature points (must satisfy ``n \\geq 2``).

# Returns
- `nodes::Vector{Float64}`: Length-`n` node vector, sorted ascending, including `±1.0`.
- `weights::Vector{Float64}`: Length-`n` weight vector aligned with `nodes`.

# Errors
- Throws `error("Lobatto needs n ≥ 2 ...")` if ``n < 2``.

# Notes
- If `n == 2`, this returns the endpoint-only rule.
"""
function gauss_lobatto_nodes_weights_float(
    n::Int
)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Lobatto needs n ≥ 2 (got n=$n)")

    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[1]   = -1.0
    t[end] =  1.0

    nn = Float64(n)
    w[1]   = 2.0 / (nn * (nn - 1.0))
    w[end] = w[1]

    if n == 2
        # just endpoints
        return t, w
    end

    # seeds: GL(n-2) as starting points for interior roots
    seeds, _ = gauss_legendre_nodes_weights_float(n-2)

    # Newton on G(x) = P_{n-2}(x) - x P_{n-1}(x)
    @inbounds for i in 1:(n-2)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pnm2, dPnm2) = _legendre_Pn_deriv(n-2, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            G  = Pnm2 - x * Pnm1
            dG = dPnm2 - (Pnm1 + x * dPnm1)

            dx = G / dG
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end

        t[i+1] = x

        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i+1] = 2.0 / (nn * (nn - 1.0) * (Pnm1*Pnm1))
    end

    p = sortperm(t)
    return t[p], w[p]
end

"""
    _GAUSS_CACHE :: Dict{Tuple{Int,Symbol,DataType}, Tuple{Vector,Vector}}

Process-local cache for single-interval Gauss-family nodes and weights on ``[-1,1]``.

# Description
This cache stores previously constructed Gauss-family rules keyed by
`(n, boundary, real_type)`.
It avoids repeated eigen-solves or Newton root-finding for identical requests.
"""
const _GAUSS_CACHE = Dict{Tuple{Int,Symbol,DataType}, Tuple{Vector,Vector}}()

"""
    _is_gauss_rule(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` is a Gauss-family rule symbol of the form `:gauss_pK`.

# Function description
This helper checks whether a rule symbol belongs to the Gauss-family naming scheme used
in this module.

# Arguments
- `rule`: Rule symbol.

# Returns
- `Bool`: `true` if `rule` starts with `"gauss_p"`, else `false`.

# Errors
- This function does not throw explicitly.
"""
@inline function _is_gauss_rule(
    rule::Symbol
)::Bool
    startswith(String(rule), "gauss_p")
end

"""
    _parse_gauss_p(
        rule::Symbol
    ) -> Int

Parse the number of Gauss points from a rule symbol of the form `:gauss_pN`.

# Function description
This helper validates the naming prefix and extracts the integer point count encoded in
`rule`.

# Arguments
- `rule`: Rule symbol expected to follow the `:gauss_p2`, `:gauss_p3`, ... pattern.

# Returns
- `Int`: Parsed number of points `n`.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if the symbol does not start with
  `"gauss_p"` or if the parsed point count is invalid.
"""
function _parse_gauss_p(
    rule::Symbol
)::Int
    _is_gauss_rule(rule) || JobLoggerTools.error_benji("rule must start with :gauss_p (got $rule)")
    s = String(rule)
    n = parse(Int, s[8:end])  # after "gauss_p"
    n >= 1 || JobLoggerTools.error_benji("gauss points must be ≥ 1 (got $n)")
    return n
end

"""
    _gauss_family_nodes_weights(
        n::Int,
        boundary::Symbol;
        real_type = nothing,
    ) -> Tuple

Return the single-interval Gauss-family nodes and weights on ``[-1,1]`` for a given
boundary selector.

# Function description
This dispatcher selects among Gauss-Legendre, left-Radau, right-Radau, and Lobatto
according to `boundary`. Results are cached in [`_GAUSS_CACHE`](@ref) and converted
to the requested scalar type.

# Arguments
- `n`: Number of points per rule.
- `boundary`: One of `:LU_EXEX`, `:LU_INEX`, `:LU_EXIN`, or `:LU_ININ`.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used for the returned nodes and weights.
  If `nothing`, `Float64` is used.

# Returns
- `nodes`: Nodes on ``[-1,1]`` in the active scalar type.
- `weights`: Weights aligned with `nodes`, in the active scalar type.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `boundary` is invalid or if `n` is
  too small for the requested family.
"""
function _gauss_family_nodes_weights(
    n::Int,
    boundary::Symbol;
    real_type = nothing,
)::Tuple

    T = isnothing(real_type) ? Float64 : real_type

    key = (n, boundary, T)
    cached = get(_GAUSS_CACHE, key, nothing)
    cached !== nothing && return cached

    t0, w0 = if boundary === :LU_EXEX
        gauss_legendre_nodes_weights_float(n)
    elseif boundary === :LU_INEX
        n >= 2 || JobLoggerTools.error_benji("Gauss-Radau requires n ≥ 2 (got n=$n)")
        gauss_radau_left_nodes_weights_float(n)
    elseif boundary === :LU_EXIN
        n >= 2 || JobLoggerTools.error_benji("Gauss-Radau requires n ≥ 2 (got n=$n)")
        gauss_radau_right_nodes_weights_float(n)
    elseif boundary === :LU_ININ
        n >= 2 || JobLoggerTools.error_benji("Gauss-Lobatto requires n ≥ 2 (got n=$n)")
        gauss_lobatto_nodes_weights_float(n)
    else
        JobLoggerTools.error_benji("boundary must be :LU_ININ|:LU_EXIN|:LU_INEX|:LU_EXEX (got $boundary)")
    end

    t = T.(t0)
    w = T.(w0)

    _GAUSS_CACHE[key] = (t, w)
    return t, w
end

"""
    _local_boundary_for_block(
        boundary::Symbol,
        blk::Int,
        N::Int
    ) -> Symbol

Translate a global boundary selection into the per-block boundary rule used in
composite Gauss quadrature.

# Function description
In the Maranatha interface, `boundary` is interpreted as a property of the whole
integration interval. This helper maps that global choice to the boundary condition
actually used on an individual composite block, so that only the true outermost blocks
receive endpoint-sensitive Radau behavior while interior blocks remain Legendre.

# Arguments
- `boundary`: Global boundary selector.
- `blk`: Zero-based block index.
- `N`: Total number of blocks.

# Returns
- `Symbol`: Per-block boundary selector.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `boundary` is invalid.
"""
@inline function _local_boundary_for_block(
    boundary::Symbol,
    blk::Int,
    N::Int
)::Symbol
    if boundary === :LU_EXEX
        return :LU_EXEX
    elseif boundary === :LU_INEX
        return (blk == 0) ? :LU_INEX : :LU_EXEX
    elseif boundary === :LU_EXIN
        return (blk == N - 1) ? :LU_EXIN : :LU_EXEX
    elseif boundary === :LU_ININ
        if blk == 0
            return :LU_INEX
        elseif blk == N - 1
            return :LU_EXIN
        else
            return :LU_EXEX
        end
    else
        JobLoggerTools.error_benji(
            "boundary must be :LU_ININ|:LU_EXIN|:LU_INEX|:LU_EXEX (got $boundary)"
        )
    end
end

"""
    _composite_gauss_nodes_weights(
        a,
        b,
        N,
        npts,
        boundary;
        real_type = nothing,
        λ = nothing,
    ) -> Tuple

Construct composite Gauss-family nodes and weights on ``[a,b]`` by repeating an
`npts`-point rule over `N` uniform subintervals.

# Function description
This helper maps a single-block Gauss-family rule from ``[-1,1]`` to each uniform block
of the global interval and concatenates all nodes and weights into flat arrays. The
boundary treatment is applied only to the true first and last blocks through
[`_local_boundary_for_block`](@ref).

# Arguments
- `a`, `b`: Global interval endpoints.
- `N`: Number of uniform blocks.
- `npts`: Number of Gauss points per block.
- `boundary`: Global boundary selector.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used internally for node and weight construction.
- `λ = nothing`:
  Reserved optional parameter forwarded for interface compatibility.
  It is currently unused by the Gauss backend.

# Returns
- `xs`: Composite node array on ``[a,b]`` in the active scalar type.
- `ws`: Composite weight array aligned with `xs`, in the active scalar type.

# Errors
- Does not validate `N` explicitly; invalid inputs may fail downstream.
- Propagates any error from [`_gauss_family_nodes_weights`](@ref).
"""
function _composite_gauss_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    npts::Int,
    boundary::Symbol;
    real_type = nothing,
    λ = nothing,
)::Tuple

    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    aa = convert(T, a)
    bb = convert(T, b)
    h  = (bb - aa) / T(N)

    xs = Vector{T}(undef, npts * N)
    ws = Vector{T}(undef, npts * N)

    half = T(0.5)
    idx = 1
    for blk in 0:(N - 1)
        local_boundary = _local_boundary_for_block(boundary, blk, N)
        t, w = _gauss_family_nodes_weights(npts, local_boundary; real_type = T)

        xL = aa + T(blk) * h
        xR = xL + h
        mid   = (xL + xR) * half
        scale = (xR - xL) * half

        @inbounds for i in 1:npts
            xs[idx] = mid + scale * t[i]
            ws[idx] = scale * w[i]
            idx += 1
        end
    end

    return xs, ws
end

"""
    _composite_gauss_u_grid(
        N,
        npts,
        boundary;
        real_type = Float64,
    ) -> Tuple

Build a dimensionless composite Gauss grid on ``u \\in [0,N]``.

# Function description
This is the dimensionless analogue of [`_composite_gauss_nodes_weights`](@ref). It maps
an `npts`-point Gauss-family rule from ``[-1,1]`` to each unit block ``[m,m+1]`` and
returns the concatenated nodes and weights on the ``u`` grid.

# Arguments
- `N`: Number of unit blocks (must satisfy ``N \\geq 1``).
- `npts`: Number of points per block.
- `boundary`: Global boundary selector.

# Keyword arguments
- `real_type = Float64`:
  Scalar type used for the returned dimensionless nodes and weights.

# Returns
- `U`: Dimensionless composite nodes on ``[0,N]`` in the active scalar type.
- `W`: Dimensionless composite weights aligned with `U`, in the active scalar type.

# Errors
- Throws `ArgumentError("N must be ≥ 1")` if ``N < 1``.
- Propagates any error from [`_gauss_family_nodes_weights`](@ref).
"""
function _composite_gauss_u_grid(
    N::Int,
    npts::Int,
    boundary::Symbol;
    real_type = Float64,
)::Tuple

    T = real_type
    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    U = Vector{T}(undef, npts * N)
    W = Vector{T}(undef, npts * N)

    half = T(0.5)
    idx = 1
    for m in 0:(N - 1)
        local_boundary = _local_boundary_for_block(boundary, m, N)
        t, w = _gauss_family_nodes_weights(npts, local_boundary; real_type = T)

        mm = T(m)
        @inbounds for i in 1:npts
            U[idx] = mm + (t[i] + one(T)) * half
            W[idx] = w[i] * half
            idx += 1
        end
    end

    return U, W
end

end  # module Gauss