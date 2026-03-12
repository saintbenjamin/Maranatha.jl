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
import ..ErrorEstimate.ErrorDispatch

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
        nerr_terms::Int = 1,
    )

Perform a weighted least-``\\chi^2`` fit for ``h \\to 0`` extrapolation from a raw
convergence dataset.

This is typically the second stage of a standard `Maranatha.jl` workflow:
generate a dataset with [`Maranatha.Runner.run_Maranatha`](@ref), fit the convergence
model with `least_chi_square_fit`, and optionally visualize the result with
[`Maranatha.PlotTools.plot_convergence_result`](@ref).

# Arguments

* `a`, `b`:
  Integration bounds, used to infer a representative subdivision count from the
  smallest step size in `hs`.
* `hs`:
  Step-size collection.
* `estimates`:
  Quadrature estimates corresponding to `hs`.
* `error_infos`:
  Error-estimator outputs, typically `result.err` from
  [`Maranatha.Runner.run_Maranatha`](@ref).
* `rule`, `boundary`:
  Rule configuration used to infer the fit powers.

# Keyword arguments

* `nterms::Int = 2`:
  Number of fit terms, including the constant extrapolated term.
* `ff_shift::Int = 0`:
  Forward shift applied when selecting fit powers from the residual-power list.
* `nerr_terms::Int = 1`:
  Number of residual contributions summed to construct the effective uncertainty vector.

# Returns

A `NamedTuple` containing the extrapolated value, parameter errors, covariance,
selected powers, and ``\\chi^2`` diagnostics.

# Notes

* This routine fits a linear-in-parameters convergence ansatz using weighted least squares.
* Residual powers are inferred automatically from the midpoint-residual model.
* The returned `powers` field is stored explicitly so that downstream plotting and
  reporting can reconstruct the fitted model consistently.

For a fuller explanation of exponent selection, covariance construction, and workflow
examples, see the `Maranatha.LeastChiSquareFit` documentation page.
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

    ks, _center = ErrorDispatch._leading_residual_ks_with_center_any(
        rule, boundary, Nref; nterms=nterms, kmax=256
    )

    # ------------------------------------------------------------
    # Select fit powers with optional forward-shift
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Map residual indices -> fit powers in h
    #
    # Convention:
    # - Newton-Cotes rules: ks already treated as powers (your current pipeline convention)
    # - Gauss / B-spline rules: ks are moment indices k, so power is (k+1)
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
    least_chi_square_fit(
        result;
        nterms::Union{Nothing,Int} = nothing,
        ff_shift::Union{Nothing,Int} = nothing,
        nerr_terms::Union{Nothing,Int} = nothing,
    )

Run [`least_chi_square_fit`](@ref) directly from a [Maranatha.Runner.run_Maranatha](@ref) result object.

If a keyword argument is omitted, the corresponding value stored in `result`
is reused.

# Arguments

* `result`:
  Result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).

# Keyword arguments

* `nterms`:
  Number of fit terms. Defaults to `result.fit_terms`.
* `ff_shift`:
  Forward shift used for fit-power selection. Defaults to `result.ff_shift`.
* `nerr_terms`:
  Number of residual terms used to build the effective uncertainty vector.
  Defaults to `result.nerr_terms`.

# Returns

The same fit-result `NamedTuple` returned by the main
[`least_chi_square_fit`](@ref) method.
"""
function least_chi_square_fit(
    result;
    nterms::Union{Nothing,Int} = nothing,
    ff_shift::Union{Nothing,Int} = nothing,
    nerr_terms::Union{Nothing,Int} = nothing,
)
    fit_nterms = isnothing(nterms) ? result.fit_terms : nterms
    fit_ff_shift = isnothing(ff_shift) ? result.ff_shift : ff_shift
    fit_nerr_terms = isnothing(nerr_terms) ? result.nerr_terms : nerr_terms

    return least_chi_square_fit(
        result.a,
        result.b,
        result.h,
        result.avg,
        result.err,
        result.rule,
        result.boundary;
        nterms = fit_nterms,
        ff_shift = fit_ff_shift,
        nerr_terms = fit_nerr_terms,
    )
end

"""
    print_fit_result(
        fit
    ) -> Nothing

Print a formatted summary of a least-``\\chi^2`` fit result.

This routine is typically called after [`least_chi_square_fit`](@ref) to display
fitted parameters, uncertainties, and fit-quality diagnostics in a compact form.

# Arguments

* `fit`:
  Fit-result object returned by [`least_chi_square_fit`](@ref).

# Returns

`nothing`.

This routine is used for its side effect: it prints a formatted summary to standard output.
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