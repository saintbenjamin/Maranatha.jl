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

import ..JobLoggerTools
import ..Quadrature.BSpline

# ------------------------------------------------------------
# Float64 midpoint residual terms for composite B-spline rules
# u ∈ [0, Nsub], center c = Nsub/2
#
# diff(k)  = ∫_0^N (u-c)^k du - Σ_i w[i]*(x[i]-c)^k
# coeff(k) = diff(k) / k!
#
# We detect nonzero residuals by a tolerance test (like Gauss path).
# ------------------------------------------------------------

"""
    _exact_moment_shifted_float(
        Nsub::Int, 
        c::Float64, 
        k::Int
    ) -> Float64

Compute the exact shifted monomial moment ``\\displaystyle{\\int\\limits_0^N \\left( u - c \\right)^k du}`` in `Float64`.

# Function description
This helper returns the closed-form shifted moment over the dimensionless
interval ``u \\in [0, N]`` with `N = Nsub`:
```math
\\int\\limits_0^{N} (u-c)^k\\,du
= \\frac{(N-c)^{k+1} - (0-c)^{k+1}}{k+1} \\,.
```

It is used by the midpoint-centered B-spline residual detector to compute the
reference exact moment in floating point.

# Arguments
- `Nsub`: Total tiling length `N` in units of the base step.
- `c`: Center shift, typically `Nsub / 2`.
- `k`: Monomial power.

# Returns
- `Float64`: Exact shifted moment computed in `Float64`.

# Errors
- No explicit validation is performed here; callers are expected to provide
  meaningful inputs.

# Notes
- This helper is intentionally `Float64`-only and mirrors the tolerance-based
  residual detection policy used for B-spline rules.
- Large `k` can cause overflow or accuracy loss in `Float64`.
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
for composite B-spline quadrature rules on ``u \\in [0, N_{\\texttt{sub}}]``.

# Function description
This routine detects midpoint-centered residual terms for B-spline quadrature
rules of the form:

- `:bspline_interp_p2`, `:bspline_interp_p3`, ... 
- `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

Using the midpoint
```math
c = \\frac{N_{\\texttt{sub}}}{2},
```
it compares the exact shifted monomial moment
```math
\\int\\limits_0^N du \\; (u-c)^k 
```
against the quadrature-induced moment
```math
\\sum_i w_i (x_i-c)^k.
```

For each detected nonzero residual,
```math
\\texttt{diff}_k
=
\\int\\limits_0^N du \\; (u-c)^k 
-
\\sum_i w_i (x_i-c)^k,
```
the routine records the factorial-scaled coefficient
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!} \\, .
```

The first `nterms` detected pairs are returned.

# Arguments
- `rule`: B-spline rule symbol.
- `boundary`: Boundary pattern used for spline knot clamping.
- `Nsub`: Dimensionless tiling length.

# Keyword arguments
- `nterms`: Number of detected nonzero residual terms to collect.
- `kmax`: Maximum moment order to scan.
- `λ`: Smoothing strength for smoothing spline rules.
- `tol_abs`: Absolute tolerance used in residual detection.
- `tol_rel`: Relative tolerance used in residual detection.

# Returns
- `ks::Vector{Int}`: Detected residual indices `k`.
- `coeffs::Vector{Float64}`: Residual coefficients `\\dfrac{\\texttt{diff}_k}{k!}``` aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1`, `kmax < 0`, or `Nsub < 1`.
- Throws if `rule` is not a recognized B-spline rule.
- Throws if `λ < 0` for a smoothing spline rule.
- Throws if fewer than `nterms` residual terms are found up to `kmax`.

# Notes
- Nodes and weights are generated on ``[0, N_{\\texttt{sub}}]`` using
  [`BSpline.bspline_nodes_weights`](@ref).
- Residual detection is tolerance-based, not exact.
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

    BSpline._is_bspline_rule(rule) || JobLoggerTools.error_benji("expected :bspline_interp_pK or :bspline_smooth_pK (got $rule)")
    p    = BSpline._parse_bspline_p(rule)
    kind = BSpline._bspline_kind(rule)  # :interp or :smooth

    # Build B-spline quadrature on [0, Nsub].
    # We set N = Nsub so the "resolution parameter" matches the dimensionless tiling length.
    a = 0.0
    b = Float64(Nsub)

    xs, ws = if kind === :interp
        BSpline.bspline_nodes_weights(a, b, Nsub, p, boundary; kind=:interp)
    else
        (λ >= 0.0) || JobLoggerTools.error_benji("λ must be ≥ 0 for smoothing spline (got λ=$λ)")
        BSpline.bspline_nodes_weights(a, b, Nsub, p, boundary; kind=:smooth, λ=λ)
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

Return detected residual indices `k` and the center tag `:mid` for B-spline rules.

# Function description
This is a thin wrapper around
[`_leading_midpoint_residual_terms_bspline_float`](@ref) that discards the
coefficient values and returns only:

- `ks`: the detected residual orders,
- `center`: the symbol `:mid`.

This matches the interface used by the higher-level residual dispatch layer.

# Arguments
- `rule`: B-spline rule symbol.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Dimensionless tiling length.

# Keyword arguments
- `nterms`: Number of residual indices to return.
- `kmax`: Maximum moment order to scan.
- `λ`: Smoothing strength for smoothing spline rules.
- `tol_abs`: Absolute tolerance for residual detection.
- `tol_rel`: Relative tolerance for residual detection.

# Returns
- `ks::Vector{Int}`: Detected residual indices.
- `center::Symbol`: Always `:mid`.

# Errors
- Propagates any error thrown by
  [`_leading_midpoint_residual_terms_bspline_float`](@ref).
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