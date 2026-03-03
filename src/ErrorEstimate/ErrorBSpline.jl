# ============================================================================
# src/ErrorEstimate/ErrorBSpline.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorBSpline

import ..Quadrature
import ..JobLoggerTools

# ------------------------------------------------------------
# Float64 midpoint residual terms for composite B-spline rules
# u ∈ [0, Nsub], center c = Nsub/2
#
# diff(k)  = ∫_0^N (u-c)^k du - Σ_i w[i]*(x[i]-c)^k
# coeff(k) = diff(k) / k!
#
# We detect nonzero residuals by a tolerance test (like GAUSS path).
# ------------------------------------------------------------

"""
    _exact_moment_shifted_float(
        Nsub::Int, 
        c::Float64, 
        k::Int
    ) -> Float64

Compute the exact shifted monomial moment ``\\displaystyle{\\int\\limits_0^N \\left( u - c \\right)^k du}`` in `Float64`.

# Function description
This helper returns the closed-form integral of a shifted monomial over the
dimensionless composite interval ``u \\in [0, N]`` where `N = Nsub`:
```math
\\int_0^{N} (u-c)^k\\,du
= \\frac{(N-c)^{k+1} - (0-c)^{k+1}}{k+1} \\,.
```

It is used by the midpoint-centered residual detector for B-spline quadrature
rules to compute the *exact* reference moment in floating-point.

# Arguments

* `Nsub`: Total tiling length `N` in units of the base step (must satisfy `Nsub```\\ge 1`` in intended usage).
* `c`:   Center shift (typically `c = Nsub/2`).
* `k`:   Monomial power (intended usage assumes ``\\texttt{k} \\ge 0``).

# Returns

* `Float64`: The exact shifted moment computed in `Float64`.

# Notes

* This routine is *Float64-only* by design, mirroring the [`Maranatha.Quadrature.Gauss`](@ref)-style tolerance-based
  residual detection path.
* For large `k`, powers can overflow or lose accuracy in `Float64`; callers should
  bound `kmax` appropriately.
"""
@inline function _exact_moment_shifted_float(
    Nsub::Int, 
    c::Float64, 
    k::Int
)::Float64
    Nf = Float64(Nsub)
    return ((Nf - c)^(k + 1) - (0.0 - c)^(k + 1)) / Float64(k + 1)
end

"""
    _leading_midpoint_residual_terms_bspline_float(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 2,
        kmax::Int = 128,
        λ::Float64 = 0.0,
        tol_abs::Float64 = 5e4 * eps(Float64),
        tol_rel::Float64 = 5e4 * eps(Float64)
    ) -> (ks::Vector{Int}, coeffs::Vector{Float64})

Collect the first `nterms` detected nonzero midpoint-shifted residual coefficients
for composite B-spline quadrature rules on ``u \\in [0, \\texttt{Nsub}]`` (`Float64`-only).

# Function description
This routine detects leading midpoint-centered residual terms for B-spline quadrature
rules:

- Interpolation spline rule: `:bspline_interp_pK`
- Smoothing spline rule:     `:bspline_smooth_pK` (requires smoothing strength ``\\lambda``)

The residual is defined using the midpoint shift ``\\displaystyle{c = \\frac{N_{\\texttt{sub}}}{2}}``:
```math
\\texttt{diff}_k = \\int\\limits_0^N du \\; (u-c)^k - \\sum_i w_i \\, \\left( x_i - c \\right)^k
```
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}
```
Nonzero residual detection uses a tolerance criterion analogous to the [`Maranatha.Quadrature.Gauss`](@ref) path:
```math
\\left\\lvert \\texttt{diff}_k \\right\\rvert > \\texttt{tol}_\\texttt{abs} + \\texttt{tol}_\\texttt{rel} \\, \\left\\lvert \\texttt{exact} \\right\\rvert
```
where
```math
\\texttt{exact} = \\int\\limits_0^N du \\; (u - c)^k \\,.
```

The first `nterms` indices `k` that pass the detection threshold are returned,
along with the corresponding scaled coefficients ``\\displaystyle{\\frac{\\texttt{diff}_k}{k!}}``.

# Arguments

* `rule`:     B-spline rule symbol (`:bspline_interp_pK` or `:bspline_smooth_pK`).
* `boundary`: Boundary pattern for the knot clamping (`:LU_ININ`, `:LU_INEX`, `:LU_EXIN`, `:LU_EXEX`).
* `Nsub`:     Dimensionless tiling length `N` (must satisfy ``N_\\texttt{sub} \\ge 1``).

# Keyword arguments

* `nterms`: Number of detected nonzero residual terms to collect (`nterms ≥ 1`).
* `kmax`:   Maximum moment order to scan (`k = 0:kmax`).
* `λ`:      Smoothing strength for `:bspline_smooth_pK` rules (``\\lambda \\ge 0``).
* `tol_abs`: Absolute tolerance used in nonzero detection.
* `tol_rel`: Relative tolerance used in nonzero detection.

# Returns

* `ks::Vector{Int}`: Detected residual indices `k` (length `nterms`).
* `coeffs::Vector{Float64}`: Residual coefficients ``\\displaystyle{\\frac{\\texttt{diff}_k}{k!}}`` (length `nterms`).

# Errors

* Throws (via [`JobLoggerTools.error_benji`](@ref)) if:

  * `nterms < 1`, `kmax < 0`, or `Nsub < 1`,
  * `rule` is not a recognized B-spline rule symbol,
  * `λ < 0` when `rule` is a smoothing rule,
  * fewer than `nterms` residuals are found up to `kmax`.

# Implementation notes

* Nodes/weights are generated on ``[0, N_\\texttt{sub}]`` by calling
  [`Quadrature.BSpline.bspline_nodes_weights`](@ref) with `N = Nsub` so the
  dimensionless length matches the composite tiling parameter.
* Factorial scaling is tracked incrementally via `inv_fact = 1/k!` updated as:
  `inv_fact /= (k+1)`. If `inv_fact` underflows to `0.0`, the scan stops early.
* For large `k`, ``(x-c)^k`` and ``(N-c)^{k+1}`` can overflow in `Float64`; use `kmax`
  conservatively if this becomes an issue.
"""
function _leading_midpoint_residual_terms_bspline_float(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 2,
    kmax::Int = 128,
    λ::Float64 = 0.0,
    tol_abs::Float64 = 5e4 * eps(Float64),
    tol_rel::Float64 = 5e4 * eps(Float64)
)::Tuple{Vector{Int}, Vector{Float64}}

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0")
    (Nsub >= 1)   || JobLoggerTools.error_benji("Nsub must be ≥ 1 (got Nsub=$Nsub)")

    Quadrature.BSpline._is_bspline_rule(rule) || JobLoggerTools.error_benji("expected :bspline_interp_pK or :bspline_smooth_pK (got $rule)")
    p    = Quadrature.BSpline._parse_bspline_p(rule)
    kind = Quadrature.BSpline._bspline_kind(rule)  # :interp or :smooth

    # Build B-spline quadrature on [0, Nsub].
    # We set N = Nsub so the "resolution parameter" matches the dimensionless tiling length.
    a = 0.0
    b = Float64(Nsub)

    xs, ws = if kind === :interp
        Quadrature.BSpline.bspline_nodes_weights(a, b, Nsub, p, boundary; kind=:interp)
    else
        (λ >= 0.0) || JobLoggerTools.error_benji("λ must be ≥ 0 for smoothing spline (got λ=$λ)")
        Quadrature.BSpline.bspline_nodes_weights(a, b, Nsub, p, boundary; kind=:smooth, λ=λ)
    end

    c = Float64(Nsub) / 2.0

    ks     = Int[]
    coeffs = Float64[]

    inv_fact = 1.0  # 1/0!
    for k in 0:kmax
        exact = _exact_moment_shifted_float(Nsub, c, k)

        approx = 0.0
        @inbounds for i in eachindex(xs)
            approx += ws[i] * (xs[i] - c)^k
        end

        diff = exact - approx

        if abs(diff) > (tol_abs + tol_rel * abs(exact))
            push!(ks, k)
            push!(coeffs, diff * inv_fact)  # diff/k!
            length(ks) == nterms && return ks, coeffs
        end

        # update inv_fact = 1/(k+1)!
        kk = k + 1
        inv_fact /= kk
        inv_fact == 0.0 && break
    end

    JobLoggerTools.error_benji("Could not collect nterms=$nterms B-spline residual terms up to kmax=$kmax (Nsub=$Nsub).")
end

"""
    _leading_residual_ks_with_center_bspline_float(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int,
        kmax::Int = 256,
        λ::Float64 = 0.0,
        tol_abs::Float64 = 5e4 * eps(Float64),
        tol_rel::Float64 = 5e4 * eps(Float64)
    ) -> (ks::Vector{Int}, center::Symbol)

Return detected residual indices `k` and the center tag `:mid` for B-spline rules (`Float64`-only).

# Function description
This is a thin wrapper around
[`_leading_midpoint_residual_terms_bspline_float`](@ref) that discards the
residual coefficient values and returns only:

- `ks`: the detected residual orders `k`
- `center`: the symbol `:mid` indicating midpoint-centered moments were used

This matches the interface of other residual-index helpers in the [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)
layer.

# Arguments
- `rule`:     B-spline rule symbol (`:bspline_interp_pK` or `:bspline_smooth_pK`).
- `boundary`: Boundary pattern symbol.
- `Nsub`:     Dimensionless tiling length (must satisfy ``N_\\texttt{sub} \\ge 1``).

# Keyword arguments
- `nterms`: Number of residual indices to return (`nterms ≥ 1`).
- `kmax`:   Maximum moment order to scan.
- `λ`:      Smoothing strength for smoothing spline rules (``\\lambda \\ge 0``).
- `tol_abs`: Absolute tolerance for residual detection.
- `tol_rel`: Relative tolerance for residual detection.

# Returns
- `ks::Vector{Int}`: Detected residual indices (length `nterms`).
- `center::Symbol`: Always `:mid`.

# Errors
- Propagates any error thrown by the underlying residual collector, including
  insufficient detected residual terms up to `kmax`.
"""
function _leading_residual_ks_with_center_bspline_float(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int,
    kmax::Int = 256,
    λ::Float64 = 0.0,
    tol_abs::Float64 = 5e4 * eps(Float64),
    tol_rel::Float64 = 5e4 * eps(Float64)
)::Tuple{Vector{Int}, Symbol}

    ks, _coeffs = _leading_midpoint_residual_terms_bspline_float(
        rule, boundary, Nsub;
        nterms=nterms, kmax=kmax, λ=λ, tol_abs=tol_abs, tol_rel=tol_rel
    )
    return ks, :mid
end

end # module ErrorBSpline