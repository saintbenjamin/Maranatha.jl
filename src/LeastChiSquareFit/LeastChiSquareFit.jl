# ============================================================================
# src/LeastChiSquareFit/LeastChiSquareFit.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module LeastChiSquareFit

import ..LinearAlgebra
import ..Statistics
import ..Printf: @sprintf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..ErrorEstimate

export least_chi_square_fit, print_fit_result

"""
    least_chi_square_fit(
        a::Real,
        b::Real,
        hs,
        estimates,
        error_infos,
        rule::Symbol,
        boundary::Symbol;
        nterms::Int = 2,
        ff_shift::Int = 0,
        nerr_terms::Int=1
    )

Perform a weighted least-``\\chi^2`` fit for ``h \\to 0`` extrapolation from a raw
convergence dataset, and return both best-fit parameters and their uncertainties.

This is typically the **second step** in a standard `Maranatha.jl` workflow:
first generate `result` with [`Maranatha.Runner.run_Maranatha`](@ref),
then call `least_chi_square_fit(result.a, result.b, result.h, result.avg, result.err, ...)`,
and optionally visualize the fitted result with
[`Maranatha.PlotTools.plot_convergence_result`](@ref).

# Function description
This routine takes a raw convergence dataset—typically produced by
[`Maranatha.Runner.run_Maranatha`](@ref)—and fits a rule- and boundary-dependent
convergence ansatz that is **linear in its parameters**, using weighted least squares (WLS).

The step sizes are supplied by the caller through `hs`, where
``\\displaystyle{h = \\frac{b-a}{N}}``.

The convergence exponents are inferred automatically from a unified midpoint-residual model
that depends on `(rule, boundary)` and a representative subdivision count `Nref`
derived from the smallest step size in `hs`.

A list of candidate residual indices is obtained via
[`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any`](@ref),
then mapped to fit powers in `h` depending on the rule family, and finally sliced using
the optional `ff_shift`.

## Exponent selection (midpoint residual model + forward shift)

First, a list of candidate residual indices is obtained via
[`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any`](@ref)
as `ks`.

These `ks` are then mapped to the fit powers in `h` depending on the rule family:

- **Newton-Cotes NS rules**: `powers_all = ks` (current pipeline convention).
- **Gauss-family rules**: `powers_all = ks` (as returned by the Gauss residual backend).
- **B-spline rules**: `powers_all = ks .+ 1` (moment index `k` corresponds to an ``h^{k+1}`` scaling).

The fit basis uses exactly `need = nterms - 1` non-constant powers, selected by slicing:

```math
\\texttt{powers} = \\texttt{powers\\_all[ (1+ff\\_shift) : (1+ff\\_shift+need-1) ]}
```
This is useful when the integrand makes the true leading-order coefficient vanish
(e.g., the corresponding midpoint derivative is identically zero), so the fit may benefit
from skipping the nominal leading power and using the next ones.

## Convergence model and design matrix

The fitted form is:

```math
I(h) = I_0 + C_1 \\, h^{p_1} + C_2 \\, h^{p_2} + \\cdots + C_{\\texttt{need}} \\, h^{p_{\\texttt{need}}}
```

with exponent list ``\\texttt{powers} = [p_1, p_2, ..., p_\\texttt{need}]`` selected as described above.

The design matrix is constructed as:

* column ``1``  : ``h^0`` (constant term)
* column ``t+1``: ``h^{\\texttt{powers[t]}}`` for ``t = 1,\\ldots,\\texttt{need}``

## Solve (WLS)

Let ``X`` be the design matrix, ``y`` the estimates, and ``\\sigma`` the pointwise errors.
Weights are constructed as ``\\displaystyle{W = \\mathrm{diag}\\left(\\frac{1}{\\sigma}\\right)}``, and the WLS problem is solved as:
```math
\\left( W \\, X \\right) \\bm{\\lambda} = \\left( W \\, y \\right),
```
yielding ``\\bm{\\lambda} = [I_0, C_1, C_2, \\ldots]^{\\mathsf{T}}``.

## Parameter covariance and errors

This implementation computes Hessian-based propagation, using a Cholesky factorization rather
than forming `inv(H)` explicitly.

1. Build the normal matrix ``A = X^{\\mathsf{T}} W^2 X``.
2. Define the ``\\chi^2`` Hessian as ``H = 2A``.
3. Factorize ``H`` via `cholesky(Symmetric(H))` (requires SPD (symmetric positive definite)).
4. Compute the covariance as:

```math
V = 4 \\, H^{-1} \\, A \\, H^{-1}.
```

The ``1\\,\\sigma`` parameter errors are ``\\sqrt{\\mathrm{diag}(V)}``.

# Arguments

* `a`, `b`:
  Integration bounds (used only to derive `Nref` from the smallest `h` in `hs`).
* `hs`:
  Vector-like collection of step sizes ``h``.
* `estimates`:
  Vector-like collection of quadrature estimates ``I(h)``.
* `error_infos`:
  Collection of error-estimator outputs, typically the `result.err` field returned by
  [`Maranatha.Runner.run_Maranatha`](@ref).
  Each entry is expected to provide residual-term information through its `terms` field.
  The first `nerr_terms` residual contributions are summed internally to build
  the effective uncertainty vector used in the weighted fit.
* `rule`:
  Quadrature rule symbol used by the midpoint residual model.
* `boundary`:
  Boundary pattern symbol (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`) used by the midpoint residual model.

# Keyword arguments

* `nterms::Int = 2`:
  Number of basis terms in the convergence model (including the constant term).
  Must satisfy ``\\texttt{nterms} \\ge 2``.

* `ff_shift::Int = 0`:
  Forward shift applied to the candidate residual-power list before selecting `nterms - 1` powers.
  Must satisfy ``\\texttt{ff\\_shift} \\ge 0``.
  If the shift makes the requested slice impossible, an error is thrown.
* `nerr_terms::Int = 1`:
  Number of midpoint residual terms used when constructing the
  effective error scale from the supplied `error_infos`.
  The first `nerr_terms` residual contributions are summed to form
  the uncertainty vector used in the weighted fit.

  In typical use, this should match the `nerr_terms` setting that was used earlier in
  [`Maranatha.Runner.run_Maranatha`](@ref), so that the fitter and the runner interpret
  the stored residual information consistently.

# Typical workflow context

A common usage pattern is:

1. generate `result` with [`Maranatha.Runner.run_Maranatha`](@ref)
2. fit the convergence model with [`least_chi_square_fit`](@ref)
3. inspect the fitted parameters with [`print_fit_result`](@ref)
4. optionally visualize the result with
   [`Maranatha.PlotTools.plot_convergence_result`](@ref)

# Returns

A `NamedTuple` with the following fields:

* `estimate::Float64`:
  Extrapolated value ``I_0 = I(h \\to 0)`` (i.e. `params[1]`).
* `error_estimate::Float64`:
  One-sigma uncertainty for ``I_0`` from the covariance diagonal.
* `params::Vector{Float64}`:
  Fitted parameter vector ``[I_0, C_1, C_2, \\ldots]``.
* `param_errors::Vector{Float64}`:
  ``1\\sigma`` uncertainties for `params`.
* `cov::Matrix{Float64}`:
  Parameter covariance matrix.
* `powers::Vector{Int}`:
  Exponent vector used by the fit basis, returned as `vcat(0, powers)` to align with `params`.
  This stored basis is later used by plotting and reporting utilities to reconstruct the
  fitted model consistently without re-inferring the powers.
* `chisq::Float64`:
  ``\\chi^2`` value.
* `redchisq::Float64`:
  ``\\chi^2/\\text{d.o.f.}`` value.
* `dof::Int`:
  Degrees of freedom, `length(y) - length(params)`.

# Errors

* Throws an error if `nterms < 2`.
* Throws an error if `ff_shift < 0`.
* Throws an error if there are not enough residual powers available after applying `ff_shift`.
* Throws an error if the Hessian is not positive definite (Cholesky decomposition fails).
* Note: If `dof == 0`, then `redchisq = chisq / dof` follows IEEE rules and may become
  `Inf` or `NaN`. In practice, this means the number of fitted parameters has saturated
  the available data points, so the fit may still return parameters but the reduced
  ``\\chi^2`` diagnostic is no longer informative.

# Example

The example below shows the standard downstream fitting step after generating
a raw convergence dataset with [`Maranatha.Runner.run_Maranatha`](@ref).

```julia
f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

print_fit_result(fit)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```
"""
function least_chi_square_fit(
    a::Real,
    b::Real,
    hs,
    estimates,
    error_infos,
    rule::Symbol,
    boundary::Symbol;
    nterms::Int=2,
    ff_shift::Int=0,
    nerr_terms::Int=1
)
    # ------------------------------------------------------------
    # Determine leading convergence power automatically
    # using composite NC residual model (midpoint expansion)
    # ------------------------------------------------------------

    Nref = round(Int, (b - a) / minimum(float.(hs)))

    ks, _center = ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any(
        rule, boundary, Nref; nterms=nterms, kmax=256
    )

    # ------------------------------------------------------------
    # Select fit powers with optional forward-shift
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Map residual indices -> fit powers in h
    #
    # Convention:
    # - NS rules: ks already treated as powers (your current pipeline convention)
    # - GAUSS / BSPLINE rules: ks are moment indices k, so power is (k+1)
    # ------------------------------------------------------------
    powers_all = if NewtonCotes._is_newton_cotes_rule(rule)
        ks
    elseif Gauss._is_gauss_rule(rule)
        if boundary === :LU_INEX || boundary === :LU_EXIN
            # Radau: shift powers so constant term is not duplicated
            # ks .+ 1
            ks
        else
            # Legendre / Lobatto
            ks
        end
    elseif BSpline._is_bspline_rule(rule)
        ks .+ 1
    else
        JobLoggerTools.error_benji("Unsupported rule family for fit-power mapping: rule=$rule")
    end

    # Defensive normalization of fit powers:
    # - Drop nonpositive powers (p <= 0) because h^0 duplicates the intercept column.
    # - Remove duplicates to avoid rank deficiency in the design matrix.
    powers_all = unique(sort(powers_all))
    powers_all = [p for p in powers_all if p > 0]

    (nterms >= 2)   || JobLoggerTools.error_benji("nterms must be >= 2 (got $nterms)")
    (ff_shift >= 0) || JobLoggerTools.error_benji("ff_shift must be ≥ 0 (got $ff_shift)")

    need  = nterms - 1
    start = 1 + ff_shift
    stop  = start + need - 1

    (stop <= length(powers_all)) || JobLoggerTools.error_benji(
        "Not enough residual powers: need $(need) terms after ff_shift=$ff_shift, but only $(length(powers_all)) available."
    )

    powers = powers_all[start:stop]

    JobLoggerTools.println_benji(
        "residual ks (backend) = [" * join(string.(ks), ", ") * "], " *
        "fit powers (h^p), ff_shift=$(ff_shift) = [" * join(string.(powers), ", ") * "]"
    )

    h = collect(float.(hs))
    y = collect(float.(estimates))

    σ = Vector{Float64}(undef, length(error_infos))

    for i in eachindex(error_infos)
        terms = error_infos[i].terms
        m = min(nerr_terms, length(terms))
        σ[i] = abs(sum(@view terms[1:m]))
    end

    any(σ .<= 0) && JobLoggerTools.error_benji(
        "Non-positive σ encountered in least_chi_square_fit"
    )

    N = length(h)

    # Design matrix:
    #   col 1: 1
    #   col t+1: h^(powers[t])  (t = 1..nterms-1)
    cols = Vector{Vector{Float64}}(undef, nterms)
    cols[1] = ones(N)

    for t in 1:need
        cols[t+1] = h .^ powers[t]
    end

    X = hcat(cols...)

    # weights
    W = LinearAlgebra.Diagonal(1.0 ./ σ)

    Xw = W * X
    yw = W * y

    # ==================================================================
    # --- WLS solve ---
    params = Xw \ yw

    # # --- covariance (Method 1) ---
    # # (Xᵀ W² X)^(-1)
    # # A = transpose(X) * (W^2) * X
    # # Cov = inv(A)
    # Cov = inv(transpose(X) * (W^2) * X)

    # param_errors = sqrt.(diag(Cov))

    # --- covariance (Method 2) ---
    # Build A = Xᵀ W² X
    A = transpose(X) * (W^2) * X

    Hess = 2.0 .* A
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Hess))  # Hess must be SPD

    # Cov = 4 * inv(Hess) * A * inv(Hess)  (computed via solves)
    M   = F \ A                            # M = inv(Hess) * A
    Cov = 4.0 .* ((F \ transpose(M))')     # Cov = 4 * M * inv(Hess)

    param_errors = sqrt.(LinearAlgebra.diag(Cov))
    # ==================================================================
    # # --- WLS solve (QR-based, numerically stable) ---
    # # Solve: minimize || W*(X*params - y) ||_2
    # # where W = LinearAlgebra.Diagonal(1 ./ σ)

    # # Weighted design and response
    # Xw = W * X
    # yw = W * y

    # # QR least squares (avoid normal equations)
    # Fqr = LinearAlgebra.qr(Xw)
    # params = Fqr \ yw

    # # Covariance of params: Cov ≈ inv(Xw'Xw)
    # # For QR: Xw = Q*R  =>  Xw'Xw = R'R  =>  inv(Xw'Xw) = inv(R)*inv(R') = inv(R'R)
    # R = Fqr.R
    # Cov = inv(transpose(R) * R)

    # param_errors = sqrt.(LinearAlgebra.diag(Cov))
    # ==================================================================

    # diagnostics
    yhat  = X * params
    resid = y .- yhat

    chisq = sum((resid ./ σ).^2)
    dof   = length(y) - length(params)
    redchisq = chisq / dof

    return (;
        estimate       = params[1],
        error_estimate = param_errors[1],
        params         = params,
        param_errors   = param_errors,
        cov            = Cov,
        powers         = vcat(0, powers), 
        chisq          = chisq,
        redchisq       = redchisq,
        dof            = dof
    )
end

"""
    print_fit_result(
        fit
    ) -> Nothing

Print a formatted summary of a least-``\\chi^2`` convergence fit.

This routine is typically called immediately after
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref) to display the fitted
parameters, uncertainties, and ``\\chi^2`` diagnostics in a compact human-readable form.

# Function description
This routine prints each fitted parameter ``\\lambda_k`` together with its
``1 \\,\\sigma`` uncertainty using
[`Maranatha.Utils.AvgErrFormatter.avgerr_e2d_from_float`](@ref).

The output includes:

* the fitted parameters ``[I_0, C_1, C_2, \\ldots]``,
* their corresponding uncertainties,
* the extrapolated value ``I_0 = I(h \\to 0)``,
* and the fit-quality diagnostics ``\\chi^2`` and ``\\chi^2/\\mathrm{d.o.f.}``.

The output formatting and ordering are intentionally kept identical to the
original implementation so that printed results remain stable across versions
and are easy to compare in logs or analysis notebooks.

# Arguments
- `fit`:
  Fit result object, typically the `NamedTuple` returned by
  [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

  The object is expected to provide at least the following fields:
    - `estimate::Float64`:
    Extrapolated value ``I_0 = I(h \to 0)`` (i.e. the first entry of `params`).
    - `error_estimate::Float64`:
    One-sigma uncertainty for ``I_0``, obtained from the covariance diagonal.
    - `params::Vector{Float64}`: Fitted parameter vector `[I0, C1, C2, ...]`.
    - `param_errors::Vector{Float64}`: ``1 \\, \\sigma`` uncertainties for `params`.
    - `cov::Matrix{Float64}`: Parameter covariance matrix.
    - `chisq::Float64`: ``\\chi^2`` value.
    - `redchisq::Float64`: ``\\chi^2/\\text{d.o.f.}``.
    - `dof::Int`: Degrees of freedom, `length(y) - length(params)`.

# Returns

`nothing`.

This routine is used for its side effect: it prints a formatted summary of the
fit result to the standard output.

# Example

```julia
result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2,3,4,5,6,7,8,9],
    rule = :gauss_p4,
    boundary = :LU_EXEX
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary
)

print_fit_result(fit)
```
"""
function print_fit_result(
    fit
)
    jobid = nothing

    for i in eachindex(fit.params)
        if !isfinite(fit.params[i]) || !isfinite(fit.param_errors[i])
            tmp_str = @sprintf("%.12e (%.12e)", fit.params[i], fit.param_errors[i])
        else
            tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.params[i], fit.param_errors[i])
        end
        JobLoggerTools.println_benji("           λ_$(i-1) = $(tmp_str)", jobid)
    end

    JobLoggerTools.println_benji("",jobid)

    JobLoggerTools.println_benji(
        @sprintf(
            "Chi^2 / d.o.f. = %.12e / %d = %.12e",
            fit.chisq,
            fit.dof,
            fit.redchisq
        ), jobid
    )

    if !isfinite(fit.estimate) || !isfinite(fit.error_estimate)
        tmp_str = @sprintf("%.12e (%.12e)", fit.estimate, fit.error_estimate)
    else
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.estimate, fit.error_estimate)
    end
    JobLoggerTools.println_benji("Result (h→0)   = $(tmp_str)", jobid)

    JobLoggerTools.println_benji("",jobid)
end

end  # module LeastChiSquareFit