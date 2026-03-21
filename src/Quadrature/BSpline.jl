# ============================================================================
# src/Quadrature/BSpline.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module BSpline

B-spline quadrature rule parser and 1D node/weight construction backend.

# Module description
`BSpline` implements the quadrature-family-specific helpers for the
interpolation and smoothing B-spline rules used by `Maranatha.Quadrature`.

Its responsibilities include:

- identifying B-spline rule symbols,
- parsing B-spline degree and variant information,
- assembling knot vectors and local basis data,
- constructing 1-dimensional nodes and weights for composite B-spline rules.

This module is a family-specific backend. Higher-level tensor-product
integration and axis-wise rule dispatch are handled elsewhere by
`QuadratureNodes` and `QuadratureDispatch`.

# Notes
- This is an internal module.
- Public quadrature entry points should normally be accessed through
  `Maranatha.Quadrature`, not by calling `BSpline` helpers directly.
"""
module BSpline

import ..LinearAlgebra

import ..JobLoggerTools

"""
    _is_bspline_rule(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` is a B-spline quadrature rule symbol.

# Function description
A rule is considered a B-spline rule if its symbol string starts with either:

- `"bspline_interp_p"` : interpolation B-spline quadrature
- `"bspline_smooth_p"` : smoothing B-spline quadrature

This helper is intended for lightweight dispatch / validation and does not
parse the degree itself.

# Arguments
- `rule`: Quadrature rule symbol to test.

# Returns
- `Bool`: `true` if `rule` matches a supported B-spline rule prefix, otherwise `false`.

# Errors
- No explicit error is thrown. Invalid or unrelated symbols simply return `false`.
"""
@inline function _is_bspline_rule(
    rule::Symbol
)::Bool
    s = String(rule)
    startswith(s, "bspline_interp_p") || startswith(s, "bspline_smooth_p")
end

"""
    _bspline_kind(
        rule::Symbol
    ) -> Symbol

Decode the B-spline rule kind from a rule symbol.

# Function description
Given a rule symbol of the form:

- `:bspline_interp_p2`, `:bspline_interp_p3`, ...
- `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

this helper returns the corresponding kind tag:

- `:interp` for interpolation spline quadrature
- `:smooth` for smoothing spline quadrature

# Arguments
- `rule`: B-spline rule symbol.

# Returns
- `Symbol`: Either `:interp` or `:smooth`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `rule` is not a recognized B-spline rule.
"""
@inline function _bspline_kind(
    rule::Symbol
)::Symbol
    s = String(rule)
    if startswith(s, "bspline_interp_p")
        return :interp
    elseif startswith(s, "bspline_smooth_p")
        return :smooth
    else
        JobLoggerTools.error_benji("Not a B-spline rule: rule=$rule")
    end
end

"""
    _parse_bspline_p(
        rule::Symbol
    ) -> Int

Parse the spline degree `p` from a B-spline rule symbol.

# Function description
This helper extracts the integer degree `p` from rule symbols:

- `:bspline_interp_p2`, `:bspline_interp_p3`, ...
- `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

where `2`, `3`, etc. are nonnegative integer degrees.

# Arguments
- `rule`: B-spline rule symbol.

# Returns
- `Int`: Parsed spline degree `p` (guaranteed ``p \\geq 0`` if successful).

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `rule` is not a B-spline rule,
  or if the parsed degree is invalid (``p < 0``).
"""
@inline function _parse_bspline_p(
    rule::Symbol
)::Int
    s = String(rule)
    if startswith(s, "bspline_interp_p")
        p = parse(Int, s[17:end])
        p >= 0 || JobLoggerTools.error_benji("B-spline degree must be ≥ 0 (got p=$p)")
        return p
    elseif startswith(s, "bspline_smooth_p")
        p = parse(Int, s[17:end])
        p >= 0 || JobLoggerTools.error_benji("B-spline degree must be ≥ 0 (got p=$p)")
        return p
    else
        JobLoggerTools.error_benji("Not a B-spline rule: rule=$rule")
    end
end

"""
    _build_knots_uniform(
        a,
        b,
        N::Int,
        p::Int,
        boundary::Symbol
    ) -> AbstractVector{<:Real}

Construct a uniform knot vector with endpoint clamping controlled by `boundary`.

# Function description
This routine builds a uniform knot line on ``[a, b]`` with step size
``h = \\dfrac{b-a}{N}`` and a simple extended knot sequence. Endpoint clamping is
then enforced according to `boundary`:

- left clamp  if `boundary ∈ { :LU_ININ, :LU_INEX }`
- right clamp if `boundary ∈ { :LU_ININ, :LU_EXIN }`

# Arguments
- `a`: Left endpoint.
- `b`: Right endpoint.
- `N`: Number of subintervals defining the uniform step.
- `p`: Spline degree.
- `boundary`: Boundary pattern (`:LU_ININ`, `:LU_INEX`, `:LU_EXIN`, `:LU_EXEX`).

# Returns
- `AbstractVector{<:Real}`:
  Knot vector `t` in the promoted scalar type of `a` and `b`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `N < 1` or `p < 0`.
"""
function _build_knots_uniform(
    a,
    b,
    N::Int,
    p::Int,
    boundary::Symbol
)
    T = promote_type(typeof(a), typeof(b))

    (N >= 1) || JobLoggerTools.error_benji("B-spline quadrature requires N ≥ 1 (got N=$N)")
    (p >= 0) || JobLoggerTools.error_benji("B-spline degree p must be ≥ 0 (got p=$p)")

    aa = convert(T, a)
    bb = convert(T, b)
    h = (bb - aa) / T(N)

    t = Vector{T}(undef, N + 2p + 1)
    @inbounds for k in 0:(N + 2p)
        t[k + 1] = (aa - T(p) * h) + T(k) * h
    end

    if boundary === :LU_ININ || boundary === :LU_INEX
        @inbounds for i in 1:(p + 1)
            t[i] = aa
        end
    end
    if boundary === :LU_ININ || boundary === :LU_EXIN
        @inbounds for i in (length(t) - p):length(t)
            t[i] = bb
        end
    end

    return t
end

"""
    _greville_points(
        t::AbstractVector,
        p::Int
    ) -> AbstractVector{<:Real}

Compute Greville abscissae for a B-spline basis defined by knots `t` and degree `p`.

# Function description
For degree ``p \\geq 1``, the Greville points are the averages of consecutive interior
knots. For the special case ``p = 0``, this routine uses knot-span midpoints.

# Arguments
- `t`: Knot vector.
- `p`: Spline degree (``p \\geq 0``).

# Returns
- `AbstractVector{<:Real}`:
  Greville points `xs` of length `nbasis`, in the element type of `t`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if the knot vector is invalid for the given `p`
  (for example, if the implied basis count is less than `1`).
"""
function _greville_points(
    t::AbstractVector,
    p::Int
)
    T = eltype(t)
    nb = length(t) - p - 1
    nb >= 1 || JobLoggerTools.error_benji("Invalid knot vector: length(t)=$(length(t)) p=$p")

    if p == 0
        xs = Vector{T}(undef, nb)
        @inbounds for i in 1:nb
            xs[i] = T(0.5) * (t[i] + t[i + 1])
        end
        return xs
    end

    xs = Vector{T}(undef, nb)
    @inbounds for i in 1:nb
        s = zero(T)
        for j in (i + 1):(i + p)
            s += t[j]
        end
        xs[i] = s / T(p)
    end
    return xs
end

"""
    _bspline_basis_all(
        x,
        t::AbstractVector,
        p::Int
    ) -> AbstractVector{<:Real}

Evaluate all B-spline basis functions ``B_{i,p}(x)`` at a point `x`.

# Function description
This routine returns the full vector of basis values for the knot vector `t`
and degree `p`. It starts from degree-``0`` indicator functions and elevates the
basis degree iteratively using the Cox-de Boor recursion.

# Arguments
- `x`: Evaluation point.
- `t`: Knot vector.
- `p`: Spline degree (``p \\geq 0``).

# Returns
- `AbstractVector{<:Real}`:
  Basis values at `x` (length `nbasis`), in the promoted scalar type of `x`
  and `t`.

# Errors
- No explicit error is thrown. Invalid inputs may instead lead to incorrect basis size
  or indexing failure upstream.
"""
function _bspline_basis_all(
    x,
    t::AbstractVector,
    p::Int
)
    T = promote_type(typeof(x), eltype(t))
    nb = length(t) - p - 1
    vals0 = zeros(T, nb)

    xx = convert(T, x)

    @inbounds for i in 1:nb
        if (t[i] <= xx < t[i + 1]) || (xx == t[end] && i == nb)
            vals0[i] = one(T)
        end
    end

    if p == 0
        return vals0
    end

    prev = vals0
    curr = zeros(T, nb)

    for k in 1:p
        fill!(curr, zero(T))
        @inbounds for i in 1:nb
            den1 = t[i + k] - t[i]
            if den1 != zero(T)
                curr[i] += (xx - t[i]) / den1 * prev[i]
            end
            if i < nb
                den2 = t[i + k + 1] - t[i + 1]
                if den2 != zero(T)
                    curr[i] += (t[i + k + 1] - xx) / den2 * prev[i + 1]
                end
            end
        end
        prev, curr = curr, prev
    end

    return prev
end

"""
    _basis_integrals(
        t::AbstractVector,
        p::Int
    ) -> AbstractVector{<:Real}

Compute the exact integral of each normalized B-spline basis function.

# Function description
For a normalized B-spline basis ``B_{i,p}(x)`` defined on knots `t`, this routine
returns the vector of basis integrals using the standard closed-form expression.

# Arguments
- `t`: Knot vector.
- `p`: Spline degree (``p \\geq 0``).

# Returns
- `AbstractVector{<:Real}`:
  Basis integrals `bI` (length `nbasis`), in the element type of `t`.

# Errors
- No explicit error is thrown. The caller is expected to provide a valid knot vector
  and degree.
"""
function _basis_integrals(
    t::AbstractVector,
    p::Int
)
    T = eltype(t)
    nb = length(t) - p - 1
    b = Vector{T}(undef, nb)
    inv = one(T) / T(p + 1)
    @inbounds for i in 1:nb
        b[i] = (t[i + p + 1] - t[i]) * inv
    end
    return b
end

"""
    _roughness_R_second_diff(
        nb::Int;
        real_type = Float64,
    ) -> AbstractMatrix{<:Real}

Construct a simple second-difference roughness penalty matrix.

# Function description
This helper builds a discrete Tikhonov-style penalty matrix based on second
differences of the spline coefficient vector. It is used as a practical
stabilizer for smoothing-mode quadrature.

# Arguments
- `nb`: Number of basis functions.

# Keyword arguments
- `real_type = Float64`:
  Scalar type used for constructing the penalty matrix.

# Returns
- `AbstractMatrix{<:Real}`:
  Symmetric `nb × nb` penalty matrix in the selected scalar type.

# Errors
- No explicit error is thrown. For `nb ≤ 2`, the routine returns a zero matrix.
"""
function _roughness_R_second_diff(
    nb::Int;
    real_type = Float64,
)
    T = real_type
    if nb <= 2
        return zeros(T, nb, nb)
    end
    D = zeros(T, nb - 2, nb)
    @inbounds for i in 1:(nb - 2)
        D[i, i]     = one(T)
        D[i, i + 1] = -T(2)
        D[i, i + 2] = one(T)
    end
    return transpose(D) * D
end

"""
    _solve_singular_safe(
        M::AbstractMatrix,
        b::AbstractVector;
        rtol = 1e-12
    ) -> AbstractVector{<:Real}

Solve a square linear system `M * x = b` robustly, with a safe fallback for
singular matrices.

# Function description
This helper first tries the standard dense solve `M \\ b`. If that fails with
`LinearAlgebra.SingularException`, it falls back to an SVD-based pseudo-inverse
solve and returns a minimum-norm solution on the retained singular subspace.

# Arguments
- `M`: Square coefficient matrix.
- `b`: Right-hand-side vector, with matching length.

# Keyword arguments
- `rtol = 1e-12`:
  Relative cutoff used in the pseudo-inverse fallback, converted to the active
  scalar type.

# Returns
- `AbstractVector{<:Real}`:
  A solution vector `x` in the promoted scalar type of `M` and `b`.

# Errors
- Re-throws any exception that is not `LinearAlgebra.SingularException`.
"""
function _solve_singular_safe(
    M::AbstractMatrix,
    b::AbstractVector;
    rtol = 1e-12
)
    T = promote_type(eltype(M), eltype(b))
    rtolT = convert(T, rtol)

    try
        return M \ b
    catch err
        if err isa LinearAlgebra.SingularException
            F = LinearAlgebra.svd(M)
            σ = F.S
            σmax = isempty(σ) ? zero(T) : maximum(σ)
            tol = rtolT * σmax

            invσ = similar(σ)
            @inbounds for i in eachindex(σ)
                invσ[i] = (σ[i] > tol) ? (one(T) / σ[i]) : zero(T)
            end

            return F.V * (invσ .* (F.U' * b))
        else
            rethrow()
        end
    end
end

"""
    bspline_nodes_weights(
        a::Real,
        b::Real,
        N::Int,
        p::Int,
        boundary::Symbol;
        kind::Symbol = :interp,
        λ = 0.0,
        real_type = nothing,
    ) -> Tuple

Construct B-spline quadrature nodes and weights on ``[a, b]``.

# Function description
This is the public entry point that produces a quadrature rule

```math
\\int_a^b dx \\; f(x) \\approx \\sum_{j=1}^{n_b} w_j \\, f(x_j)
```

where the nodes are Greville abscissae derived from a uniform knot vector and
the weights are obtained from exact basis integrals together with either spline
interpolation or a smoothing-style normal equation.

Two modes are supported:

- `kind == :interp` : interpolation-based spline quadrature
- `kind == :smooth` : smoothing-based spline quadrature with penalty `λ`

# Arguments
- `a`: Left endpoint.
- `b`: Right endpoint. Must satisfy ``b > a``.
- `N`: Number of uniform subintervals defining knot spacing.
- `p`: B-spline degree (``p \\geq 0``).
- `boundary`: Boundary pattern controlling endpoint clamping.

# Keyword arguments
- `kind`: Either `:interp` or `:smooth`.
- `λ = 0.0`:
  Smoothing strength, used only when `kind == :smooth`.
- `real_type = nothing`:
  Optional scalar type used internally for knot construction, basis evaluation,
  linear solves, nodes, and weights.

# Returns
- `xs`: Quadrature nodes (Greville points), length `nbasis`, in the active scalar type.
- `ws`: Quadrature weights, length `nbasis`, in the active scalar type.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if:
  - `b ≤ a`,
  - `N < 1`,
  - `p < 0`,
  - `boundary !== :LU_ININ`,
  - `λ < 0` in smoothing mode,
  - or the implied basis size is invalid.
"""
function bspline_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    p::Int,
    boundary::Symbol;
    kind::Symbol = :interp,
    λ = 0.0,
    real_type = nothing,
)::Tuple

    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    aa = convert(T, a)
    bb = convert(T, b)
    λT = convert(T, λ)

    (bb > aa) || JobLoggerTools.error_benji("Need b > a (got a=$a, b=$b)")
    (N >= 1)  || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (p >= 0)  || JobLoggerTools.error_benji("Need p ≥ 0 (got p=$p)")

    if boundary !== :LU_ININ
        JobLoggerTools.error_benji(
            "B-spline quadrature currently supports only boundary=:LU_ININ (clamped). " *
            "Got boundary=$boundary."
        )
    end

    t  = _build_knots_uniform(aa, bb, N, p, boundary)
    nb = length(t) - p - 1
    nb >= 1 || JobLoggerTools.error_benji("Invalid (N,p) produced nbasis=$nb")

    xs = _greville_points(t, p)
    bI = _basis_integrals(t, p)

    A = Matrix{T}(undef, nb, nb)
    @inbounds for j in 1:nb
        Bj = _bspline_basis_all(xs[j], t, p)
        @inbounds for i in 1:nb
            A[j, i] = Bj[i]
        end
    end

    if kind === :interp
        At = transpose(A)
        w  = _solve_singular_safe(At, bI; rtol = T(1e-12))
        return xs, w
    elseif kind === :smooth
        (λT >= zero(T)) || JobLoggerTools.error_benji("Smoothing λ must be ≥ 0 (got λ=$λ)")
        R = _roughness_R_second_diff(nb; real_type = T)

        M = transpose(A) * A + λT * R
        z = _solve_singular_safe(transpose(M), bI; rtol = T(1e-12))
        w = A * z
        return xs, w
    else
        JobLoggerTools.error_benji("Unknown bspline kind=$kind (use :interp or :smooth)")
    end
end

end  # module BSpline
