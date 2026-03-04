# ============================================================================
# src/Quadrature/BSpline.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BSpline

using ..LinearAlgebra

using ..JobLoggerTools

# ============================================================
# Rule symbol parsing
#   :bspline_interp_pK  -> interpolation spline quadrature (degree K)
#   :bspline_smooth_pK  -> smoothing spline quadrature (degree K)
# ============================================================

"""
    _is_bspline_rule(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` is a B-spline quadrature rule symbol.

# Function description
A rule is considered a B-spline rule if its symbol string starts with either:

- `"bspline_interp_p"` : interpolation B-spline quadrature (degree `p`)
- `"bspline_smooth_p"` : smoothing    B-spline quadrature (degree `p`)

This helper is used only for lightweight dispatch / validation and does not
parse the degree itself.

# Arguments
- `rule`: Quadrature rule symbol to test.

# Returns
- `Bool`: `true` if `rule` matches a supported B-spline rule prefix, otherwise `false`.

# Notes
- This function performs only prefix checks; it does not validate that the suffix
  is a valid integer degree.
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

- `:bspline_interp_pK`  (interpolation)
- `:bspline_smooth_pK`  (smoothing)

this helper returns the corresponding kind tag:

- `:interp` for interpolation spline quadrature (`bsplI`)
- `:smooth` for smoothing spline quadrature (`bsplS`)

# Arguments
- `rule`: B-spline rule symbol.

# Returns
- `Symbol`: Either `:interp` or `:smooth`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `rule` is not a recognized B-spline rule.

# Notes
- This function does not parse the degree `p`; use [`_parse_bspline_p`](@ref) for that.
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

- `:bspline_interp_pK`  (interpolation)
- `:bspline_smooth_pK`  (smoothing)

where `K` is a nonnegative integer degree (e.g., `0, 1, 2, 3, ...`).

# Arguments
- `rule`: B-spline rule symbol.

# Returns
- `Int`: Parsed spline degree `p` (guaranteed `p ≥ 0` if successful).

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a B-spline rule,
  or if the parsed degree is invalid (`p < 0`).
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

# ============================================================
# Knot builder (uniform, with boundary pattern via endpoint clamping)
#
# We keep knot length = N + 2p + 1 (like classic open-uniform-with-extension),
# then enforce endpoint clamping by overwriting the first/last p+1 knots if closed.
#
# boundary patterns:
#   LU_ININ: clamp both
#   LU_INEX: clamp left only
#   LU_EXIN: clamp right only
#   LU_EXEX: clamp none (fully open / extended)
# ============================================================

"""
    _build_knots_uniform(
        a::Float64, 
        b::Float64, 
        N::Int, 
        p::Int, 
        boundary::Symbol
    ) -> Vector{Float64}

Construct a uniform knot vector with endpoint clamping controlled by `boundary`.

# Function description
This routine builds a uniform knot line on ``[a, b]`` with step size:
```math
h = \\frac{b-a}{N}
```
and an extended open-uniform-style knot sequence of length:
```math
N + 2p + 1
```
The base (extended) knots are:
```math
a - p \\, h \\;,\\;  a - (p-1) \\, h \\;,\\;  \\ldots \\;,\\; b + (p-1) \\, h \\;,\\; b + p \\, h 
```
Then endpoint clamping is enforced by overwriting knots near the endpoints:

* left clamp  (repeat `a` for `p+1` knots) if `boundary ∈ { :LU_ININ, :LU_INEX }`
* right clamp (repeat `b` for `p+1` knots) if `boundary ∈ { :LU_ININ, :LU_EXIN }`

This provides a simple, deterministic boundary behavior compatible with the
Greville-point quadrature construction in this module.

# Arguments

* `a`: Left endpoint (`Float64`).
* `b`: Right endpoint (`Float64`).
* `N`: Number of subintervals defining the uniform step (``N \\ge 1``).
* `p`: Spline degree (``p \\ge 0``).
* `boundary`: Boundary pattern (`:LU_ININ`, `:LU_INEX`, `:LU_EXIN`, `:LU_EXEX`).

# Returns

* `Vector{Float64}`: Knot vector `t`.

# Errors

* Throws (via `JobLoggerTools.error_benji`) if `N < 1` or `p < 0`.

# Notes

* This is a pragmatic knot builder for quadrature, not a general-purpose spline API.
* For `:LU_EXEX` no clamping is applied (fully extended endpoints).
"""
function _build_knots_uniform(
    a::Float64,
    b::Float64,
    N::Int,
    p::Int,
    boundary::Symbol
)::Vector{Float64}

    (N >= 1) || JobLoggerTools.error_benji("B-spline quadrature requires N ≥ 1 (got N=$N)")
    (p >= 0) || JobLoggerTools.error_benji("B-spline degree p must be ≥ 0 (got p=$p)")

    h = (b - a) / Float64(N)

    # base extended knot line: a - p*h, ..., b + p*h  (step h)
    # length = (N + 2p) + 1 = N + 2p + 1
    t = Vector{Float64}(undef, N + 2p + 1)
    @inbounds for k in 0:(N + 2p)
        t[k+1] = (a - Float64(p)*h) + Float64(k)*h
    end

    # apply endpoint clamping depending on boundary
    # "closed" => repeat endpoint p+1 times by setting the first/last p+1 knots equal
    if boundary === :LU_ININ || boundary === :LU_INEX
        @inbounds for i in 1:(p+1)
            t[i] = a
        end
    end
    if boundary === :LU_ININ || boundary === :LU_EXIN
        @inbounds for i in (length(t)-p):length(t)
            t[i] = b
        end
    end

    return t
end

# ============================================================
# Greville abscissae:
#   ξ_i = (t_{i+1} + ... + t_{i+p}) / p, for i=1..nbasis
# with nbasis = length(t) - p - 1
# ============================================================

"""
    _greville_points(
        t::Vector{Float64}, 
        p::Int
    ) -> Vector{Float64}

Compute Greville abscissae for a B-spline basis defined by knots ``t`` and degree ``p``.

# Function description
For degree ``p \\ge 1``, the Greville points are defined by:
```math
\\xi_i = \\frac{1}{p}\\sum_{j=1}^{p} t_{i+j}, \\qquad i = 1,\\ldots, n_b
```
where the number of basis functions is:
```math
n_b = \\mathrm{length}(t) - p - 1 \\,.
```

For the special case ``p = 0`` (piecewise-constant splines), this routine chooses
simple knot-span midpoints:

```math
\\xi_i = \\frac{1}{2}(t_i + t_{i+1})
```

# Arguments

* `t`: Knot vector.
* `p`: Spline degree (`p ≥ 0`).

# Returns

* `Vector{Float64}`: Greville points `xs` of length `nbasis`.

# Errors

* Throws (via [`JobLoggerTools.error_benji`](@ref)) if the knot vector is invalid for the given `p`
  (e.g., `nbasis < 1`).

# Notes

* Greville points are a standard, stable node choice for spline collocation.
"""
function _greville_points(
    t::Vector{Float64}, 
    p::Int
)::Vector{Float64}
    nb = length(t) - p - 1
    nb >= 1 || JobLoggerTools.error_benji("Invalid knot vector: length(t)=$(length(t)) p=$p")

    if p == 0
        # piecewise-constant: pick midpoints of knot spans for a simple node choice
        xs = Vector{Float64}(undef, nb)
        @inbounds for i in 1:nb
            xs[i] = 0.5 * (t[i] + t[i+1])
        end
        return xs
    end

    xs = Vector{Float64}(undef, nb)
    @inbounds for i in 1:nb
        s = 0.0
        for j in (i+1):(i+p)
            s += t[j]
        end
        xs[i] = s / Float64(p)
    end
    return xs
end

# ============================================================
# B-spline basis evaluation (Cox–de Boor, stable DP)
# Returns all basis values {B_{i,p}(x)} for i=1..nbasis.
# ============================================================

"""
    _bspline_basis_all(
        x::Float64, 
        t::Vector{Float64}, 
        p::Int
    ) -> Vector{Float64}

Evaluate all B-spline basis functions ``B_{i,p}(x)`` at a point ``x`` using Cox-de Boor recursion.

# Function description
This routine returns the full vector of basis values:
```math
\\{ B_{1,p}(x), B_{2,p}(x), \\ldots, B_{n_b,p}(x) \\}
```
where:
```math
n_b = \\texttt{length(}t\\texttt{)} - p - 1
```

## Implementation details:

* Start from degree-``0`` indicator functions on knot spans ``[t_i, t_{i+1})``.
* Elevate degree iteratively via the Cox-de Boor recursion.
* Handles the endpoint convention `x == t[end]` by assigning the last basis to ``1``
  (consistent with half-open span semantics).

# Arguments

* `x`: Evaluation point.
* `t`: Knot vector.
* `p`: Spline degree (``p \\ge 0``).

# Returns

* `Vector{Float64}`: Basis values at ``x`` (length `nbasis`).

# Notes

* This routine is designed for quadrature weight construction, so it prioritizes
  deterministic behavior and numerical stability over maximum generality.
* For repeated evaluation at many ``x``, consider reusing allocations externally if needed.
"""
function _bspline_basis_all(
    x::Float64, 
    t::Vector{Float64}, 
    p::Int
)::Vector{Float64}
    nb = length(t) - p - 1
    vals0 = zeros(Float64, nb)

    # degree 0
    @inbounds for i in 1:nb
        # support [t[i], t[i+1]) except special-case x==t[end] -> last basis
        if (t[i] <= x < t[i+1]) || (x == t[end] && i == nb)
            vals0[i] = 1.0
        end
    end

    if p == 0
        return vals0
    end

    prev = vals0
    curr = zeros(Float64, nb)

    # elevate degree step-by-step
    for k in 1:p
        fill!(curr, 0.0)
        @inbounds for i in 1:nb
            # left term
            den1 = t[i+k] - t[i]
            if den1 != 0.0
                curr[i] += (x - t[i]) / den1 * prev[i]
            end
            # right term uses prev[i+1]
            if i < nb
                den2 = t[i+k+1] - t[i+1]
                if den2 != 0.0
                    curr[i] += (t[i+k+1] - x) / den2 * prev[i+1]
                end
            end
        end
        prev, curr = curr, prev
    end

    return prev
end

# ============================================================
# Exact integral of normalized B-spline basis:
#   ∫ B_{i,p}(x) dx = (t_{i+p+1} - t_i) / (p+1)
# ============================================================

"""
    _basis_integrals(
        t::Vector{Float64}, 
        p::Int
    ) -> Vector{Float64}

Compute the exact integral of each normalized B-spline basis function over ``x``.

# Function description
For a normalized B-spline basis ``B_{i,p}(x)`` defined on knots ``t``, the integral satisfies:
```math
\\int dx \\; B_{i,p}(x) = \\frac{t_{i+p+1} - t_i}{p+1}
```

This routine returns the vector:

```math
b_i = \\frac{t_{i+p+1} - t_i}{p+1}, \\qquad i = 1,\\ldots,n_b
```

where:

```math
n_b = \\texttt{length(}t\\texttt{)} - p - 1
```

# Arguments

* `t`: Knot vector.
* `p`: Spline degree (`p ≥ 0`).

# Returns

* `Vector{Float64}`: Basis integrals `bI` (length `nbasis`).

# Notes

* This formula assumes the standard normalized B-spline basis consistent with
  Cox–de Boor recursion.
"""
function _basis_integrals(
    t::Vector{Float64}, 
    p::Int
)::Vector{Float64}
    nb = length(t) - p - 1
    b = Vector{Float64}(undef, nb)
    inv = 1.0 / Float64(p + 1)
    @inbounds for i in 1:nb
        b[i] = (t[i + p + 1] - t[i]) * inv
    end
    return b
end

# ============================================================
# Smoothing penalty matrix (simple second-difference Tikhonov):
#   R = D'D where D maps coeffs -> second differences
# This is NOT the exact ∫(s''(x))^2 dx penalty, but works as a practical smoother.
# ============================================================

"""
    _roughness_R_second_diff(
        nb::Int
    ) -> Matrix{Float64}

Construct a simple second-difference roughness penalty matrix ``R = D^\\prime \\, D``.

# Function description
This helper builds a discrete Tikhonov-style penalty intended for smoothing:

- Let ``c`` be the coefficient vector (length `nb`).
- Define second differences (for interior indices):

```math
\\left( D \\, c \\right)_i = c_i - 2 \\, c_{i+1} + c_{i+2}
```

Then the quadratic penalty is:
```math
c^T R c = \\lvert D c \\rvert_2^2
```

This is **not** the exact continuous spline penalty ``\\int dx \\; \\left( s^{\\prime\\prime}(x) \\right)^2``,
but it acts as a practical stabilizer for spline smoothing in this quadrature
weight construction.

# Arguments

* `nb`: Number of basis functions (`nb```\\ge 1``).

# Returns

* `Matrix{Float64}`: Symmetric `nb```\\times```nb` penalty matrix.

# Notes

* For `nb```\\le 2``, the second-difference operator is empty and this returns zeros.
"""
function _roughness_R_second_diff(
    nb::Int
)::Matrix{Float64}
    if nb <= 2
        return zeros(Float64, nb, nb)
    end
    D = zeros(Float64, nb - 2, nb)
    @inbounds for i in 1:(nb - 2)
        D[i, i]     = 1.0
        D[i, i+1]   = -2.0
        D[i, i+2]   = 1.0
    end
    return transpose(D) * D
end

# ============================================================
# Public: composite B-spline quadrature nodes/weights on [a,b]
#
# Returns nodes xs (Greville points) and weights ws such that:
#   ∫_a^b f(x) dx  ≈  Σ ws[j] * f(xs[j])
#
# For interpolation:
#   A c = y,  I = b'c = (A' \ b)' y
# For smoothing:
#   (A'A + λR) c = A' y,  I = b'c = (A*z)' y with (A'A+λR)' z = b
#   => w = A*z
# ============================================================

"""
    bspline_nodes_weights(
        a::Real, 
        b::Real, 
        N::Int, 
        p::Int, 
        boundary::Symbol; 
        kind::Symbol=:interp, 
        λ::Float64=0.0
    ) -> (xs::Vector{Float64}, ws::Vector{Float64})

Construct B-spline quadrature nodes and weights on ``[a, b]``.

# Function description
This is the public entry point that produces a quadrature rule:
```math
\\int\\limits_a^b dx \\; f(x) \\approx \\sum_{j=1}^{n_b} w_j \\, f(x_j)
```
where:

* ``x_j`` are Greville abscissae derived from a uniform knot vector,
* ``w_j`` are weights derived from integrating the spline basis exactly
  and mapping that to a weighted sum over ``f(x_j)``.

The procedure:

1. Build a uniform knot vector `t` with boundary clamping controlled by `boundary`.
2. Compute Greville nodes `xs`.
3. Build the collocation matrix ``A_{ji} = B_{i,p} \\, \\texttt{xs}_j``.
4. Compute basis integrals ``\\texttt{bI}_i = \\int dx \\; B_{i,p}(x)`` (exact formula).
5. Derive weights `ws` depending on `kind`:

## Interpolation mode (`kind == :interp`)

Assume spline interpolation ``A \\, c = y``, so the integral is:
```math
I = b^T c = b^T A^{-1} y = (A^{-T} b)^T y \\,.
```
Thus weights are:
```math
w = A^{-T} b
```
implemented as `transpose(A) \\ bI`.

## Smoothing mode (`kind == :smooth`)

Use a Tikhonov-regularized normal equation:
```math
(A^T A + \\lambda R) c = A^T y
```
Then:
```math
I = b^T c = b^T (A^T A + \\lambda R)^{-1} A^T y = (A z)^T y
```
where ``z`` solves:
```math
(A^T A + \\lambda R)^T z = b
```
Thus weights are computed as ``w = A \\, z``.

# Arguments

* `a`: Left endpoint (real).
* `b`: Right endpoint (real), must satisfy `b > a`.
* `N`: Number of uniform subintervals defining knot spacing (`N ≥ 1`).
* `p`: B-spline degree (`p ≥ 0`).
* `boundary`: Boundary pattern controlling endpoint clamping (`:LU_ININ`, `:LU_INEX`, `:LU_EXIN`, `:LU_EXEX`).

# Keyword arguments

* `kind`: `:interp` (default) or `:smooth`.
* `λ`: Smoothing strength (`λ ≥ 0`), used only when `kind == :smooth`.

# Returns

* `xs::Vector{Float64}`: Quadrature nodes (Greville points), length `nbasis`.
* `ws::Vector{Float64}`: Quadrature weights, length `nbasis`.

# Errors

* Throws (via [`JobLoggerTools.error_benji`](@ref)) if:

  * `b ≤ a`,
  * `N < 1`,
  * `p < 0`,
  * `λ < 0` in smoothing mode,
  * or the implied basis size is invalid.

# Notes

* This routine constructs a deterministic spline-based quadrature rule; it is not adaptive.
* The smoothing penalty uses a discrete second-difference model (see [`_roughness_R_second_diff`](@ref)).
"""
function bspline_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    p::Int,
    boundary::Symbol;
    kind::Symbol = :interp,
    λ::Float64 = 0.0
)::Tuple{Vector{Float64}, Vector{Float64}}

    aa = Float64(a)
    bb = Float64(b)

    (bb > aa) || JobLoggerTools.error_benji("Need b > a (got a=$a, b=$b)")
    (N >= 1)  || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (p >= 0)  || JobLoggerTools.error_benji("Need p ≥ 0 (got p=$p)")

    # knots + basis count
    t  = _build_knots_uniform(aa, bb, N, p, boundary)
    nb = length(t) - p - 1
    nb >= 1 || JobLoggerTools.error_benji("Invalid (N,p) produced nbasis=$nb")

    xs = _greville_points(t, p)
    bI = _basis_integrals(t, p)

    # build interpolation/smoothing matrix A[j,i] = B_i(xs[j])
    A = Matrix{Float64}(undef, nb, nb)
    @inbounds for j in 1:nb
        Bj = _bspline_basis_all(xs[j], t, p)
        @inbounds for i in 1:nb
            A[j, i] = Bj[i]
        end
    end

    if kind === :interp
        # weights w = A' \ b
        w = transpose(A) \ bI
        return xs, w
    elseif kind === :smooth
        (λ >= 0.0) || JobLoggerTools.error_benji("Smoothing λ must be ≥ 0 (got λ=$λ)")
        R = _roughness_R_second_diff(nb)

        # solve (A'A + λR)' z = b, then w = A*z
        M = transpose(A) * A + λ * R
        z = transpose(M) \ bI
        w = A * z
        return xs, w
    else
        JobLoggerTools.error_benji("Unknown bspline kind=$kind (use :interp or :smooth)")
    end
end

end  # module BSpline