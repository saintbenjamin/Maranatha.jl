# ============================================================================
# src/LeastChiSquareFit/LeastChiSquareFit.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module LeastChiSquareFit

Weighted least-``\\chi^2`` fitting utilities for convergence extrapolation in
`Maranatha.jl`.

# Module description
`Maranatha.LeastChiSquareFit` converts the sampled convergence data generated
by [`Maranatha.Runner.run_Maranatha`](@ref) into a linear weighted least-squares
fit in powers of the scalarized step-size proxy `h`.

The fitter supports both derivative-based and refinement-based error metadata.
When axis-wise `rule` / `boundary` specifications are present, the candidate
fit powers are inferred from the union of the leading residual powers detected
on each axis.

# Main entry points
- [`least_chi_square_fit`](@ref)
- [`print_fit_result`](@ref)

# Notes
- This module performs fitting only; quadrature evaluation and error
  estimation happen elsewhere.
- The returned fit metadata are designed so that plotting and reporting layers
  can reconstruct the fitted model without recomputing exponent selection.
"""
module LeastChiSquareFit

import ..LinearAlgebra
import ..Statistics
import ..Printf: @sprintf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter
import ..Utils.QuadratureBoundarySpec
import ..Quadrature.QuadratureRuleSpec
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative

"""
    least_chi_square_fit(
        result;
        fit_func_terms::Int = result.fit_terms,
        ff_shift::Int = result.ff_shift,
        nerr_terms::Int = result.nerr_terms,
    )

Perform a weighted least-``\\chi^2`` fit for ``h \\to 0`` extrapolation directly
from a [`Maranatha.Runner.run_Maranatha`](@ref) result object.

This is typically the second stage of a standard `Maranatha.jl` workflow:
generate a dataset with [`Maranatha.Runner.run_Maranatha`](@ref), fit the convergence
model with `least_chi_square_fit`, and optionally visualize the result with
[`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref).

The uncertainty model used in the weighted fit is extracted from `result.err`,
which may come either from residual-based estimators (typically exposing
multiple `:terms`) or from refinement-based estimators (typically exposing a
single `:estimate` field).

# Arguments

* `result`:
  Result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).

  The fitter operates on the scalar step-size sequence `result.h`, the
  corresponding estimates `result.avg`, and the error objects `result.err`.
  The rule configuration is taken from `result.rule` and `result.boundary`.
  These may be either scalar shared specs or axis-wise tuple/vector specs.

  A representative subdivision count `Nref` is obtained directly from
  `result.nsamples` (using its minimum value), rather than being inferred
  geometrically from the integration bounds.

# Keyword arguments

* `fit_func_terms::Int = result.fit_terms`:
  Number of fit terms, including the constant extrapolated term.
* `ff_shift::Int = result.ff_shift`:
  Forward shift applied when selecting fit powers from the residual-power list.
* `nerr_terms::Int = result.nerr_terms`:
  Number of residual contributions summed to construct the effective uncertainty vector.

# Returns

A `NamedTuple` containing the extrapolated value, parameter errors, covariance,
selected powers, and ``\\chi^2`` diagnostics.

# Notes

* This routine fits a linear-in-parameters convergence ansatz using weighted least squares.
* Residual powers are inferred automatically from the midpoint-residual model.
* For axis-wise `rule` / `boundary` configurations, the candidate residual
  powers are collected on each axis and merged before `ff_shift` is applied.
* When Newton-Cotes rules are active, the representative `Nref` is adjusted to
  a common admissible subdivision count before residual-power detection.
* The returned `powers` field is stored explicitly so that downstream plotting and
  reporting can reconstruct the fitted model consistently.
* The effective uncertainty vector `σ` is constructed from backend-specific
  error-info objects through [`_extract_sigma_from_error_info`](@ref), allowing
  both residual-based and refinement-based estimators to feed the same fitter.
* The fit operates on the scalar step-size sequence stored in `result.h`.
* For a fuller explanation of exponent selection, covariance construction, and workflow
  examples, see the `Maranatha.LeastChiSquareFit` documentation page.
"""
function least_chi_square_fit(
    result;
    fit_func_terms::Int = result.fit_terms,
    ff_shift::Int = result.ff_shift,
    nerr_terms::Int = result.nerr_terms
)
    hs = result.h
    estimates = result.avg
    error_infos = result.err
    rule = result.rule
    boundary = result.boundary

    dim = Int(result.dim)

    QuadratureRuleSpec._validate_rule_spec(rule, dim)
    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)

    rule_axes = [QuadratureRuleSpec._rule_at(rule, d, dim) for d in 1:dim]
    boundary_axes = [QuadratureBoundarySpec._boundary_at(boundary, d, dim) for d in 1:dim]

    Nref = minimum(result.nsamples)

    Ncand = Nref
    while true
        updated = false

        for d in 1:dim
            rd = rule_axes[d]
            if NewtonCotes._is_newton_cotes_rule(rd)
                p = NewtonCotes._parse_newton_p(rd)
                Nd = NewtonCotes._next_valid_Nsub(p, boundary_axes[d], Ncand)
                if Nd > Ncand
                    Ncand = Nd
                    updated = true
                end
            end
        end

        updated || break
    end
    Nref = Ncand

    nks_needed = max(fit_func_terms + ff_shift - 1, 1)

    ks_lists = Vector{Vector{Int}}(undef, dim)
    for d in 1:dim
        ks_lists[d], _ = ErrorDispatchDerivative._leading_residual_ks_with_center_any(
            rule_axes[d],
            boundary_axes[d],
            Nref;
            fit_func_terms = nks_needed,
            kmax = 256,
        )
    end

    powers_all = unique(sort(vcat(ks_lists...)))
    powers_all = unique(sort(powers_all))
    powers_all = [p for p in powers_all if p > 0]

    (fit_func_terms >= 2)   || JobLoggerTools.error_benji(
        "fit_func_terms must be >= 2 (got $fit_func_terms)"
    )
    (ff_shift >= 0) || JobLoggerTools.error_benji(
        "ff_shift must be ≥ 0 (got $ff_shift)"
    )

    need  = fit_func_terms - 1
    start = 1 + ff_shift
    stop  = start + need - 1

    (stop <= length(powers_all)) || JobLoggerTools.error_benji(
        "Not enough residual powers: need $(need) terms after ff_shift=$ff_shift, but only $(length(powers_all)) available."
    )

    powers = powers_all[start:stop]

    JobLoggerTools.println_benji(
        "axiswise residual ks = " * string(ks_lists) * ", " *
        "merged ks = [" * join(string.(powers_all), ", ") * "], " *
        "fit powers (h^p), ff_shift=$(ff_shift) = [" * join(string.(powers), ", ") * "]"
    )

    h = collect(float.(hs))
    y = collect(float.(estimates))

    σ = similar(h)

    for i in eachindex(error_infos)
        σ[i] = _extract_sigma_from_error_info(
            error_infos[i];
            nerr_terms = nerr_terms,
        )
    end

    bad_fatal = [(i, s) for (i, s) in pairs(σ) if !isfinite(s) || s < zero(s)]
    if !isempty(bad_fatal)
        JobLoggerTools.warn_benji("σ (full array) = $(σ)")
        JobLoggerTools.warn_benji("invalid (index, value) = $(bad_fatal)")
        JobLoggerTools.error_benji(
            "Non-finite or negative σ encountered in least_chi_square_fit"
        )
    end

    zero_idx = [i for (i, s) in pairs(σ) if s == zero(s)]
    use_mask = trues(length(σ))

    if !isempty(zero_idx)
        use_mask[zero_idx] .= false
    end

    if !all(use_mask)
        h = h[use_mask]
        y = y[use_mask]
        σ = σ[use_mask]
    end

    N = length(h)
    T = eltype(h)

    cols = Vector{Vector{T}}(undef, fit_func_terms)
    cols[1] = ones(T, N)

    for t in 1:need
        cols[t+1] = h .^ powers[t]
    end

    X = hcat(cols...)
    W = LinearAlgebra.Diagonal(one(T) ./ σ)

    Xw = W * X
    yw = W * y

    params = Xw \ yw

    A = transpose(X) * (W^2) * X

    Hess = (one(T) + one(T)) .* A
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Hess))

    M   = F \ A
    Cov = (one(T) + one(T)) .* ((F \ transpose(M))')

    param_errors = sqrt.(LinearAlgebra.diag(Cov))

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
        fit_func_terms = fit_func_terms,
        nerr_terms     = nerr_terms,
        cov            = Cov,
        powers         = vcat(zero(T), powers),
        chisq          = chisq,
        redchisq       = redchisq,
        dof            = dof
    )
end


"""
    _extract_sigma_from_error_info(
        e;
        nerr_terms::Int = 1,
    ) -> Real

Extract an effective scalar uncertainty `σ` from an error-estimator result
object.

# Function description
This helper converts a backend-specific error-estimation result into the single
effective uncertainty value used by the weighted least-``\\chi^2`` fitter.

It supports two currently used error-info layouts:

- residual-based estimators exposing a `:terms` field, where the effective
  uncertainty is formed from the sum of the first `nerr_terms` entries, and
- refinement-based estimators exposing an `:estimate` field, where that value
  is used directly.

In both cases, the returned value is wrapped in `abs(...)` so that the fitter
always receives a nonnegative uncertainty scale.

# Arguments
- `e`:
  Error-estimator result object, typically one entry of `result.err` returned by
  [`Maranatha.Runner.run_Maranatha`](@ref).

# Keyword arguments
- `nerr_terms::Int = 1`:
  Number of leading residual terms to combine when `e` exposes a `:terms` field.

# Returns
- `Real`:
  Effective scalar uncertainty used in weighted least squares.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `e` has neither a `:terms` field
  nor an `:estimate` field.

# Notes
- For residual-based estimators, this helper uses
  `abs(sum(terms[1:m]))` with `m = min(nerr_terms, length(terms))`.
- For refinement-based estimators, this helper uses `abs(e.estimate)`.
- This function does not itself check whether the returned value is finite or
  strictly positive; that validation is performed later by
  [`least_chi_square_fit`](@ref).
"""
@inline function _extract_sigma_from_error_info(
    e;
    nerr_terms::Int = 1,
)
    if hasproperty(e, :terms)
        terms = e.terms
        m = min(nerr_terms, length(terms))
        return abs(sum(@view terms[1:m]))
    elseif hasproperty(e, :estimate)
        return abs(e.estimate)
    else
        JobLoggerTools.error_benji(
            "Unsupported error-info structure in least_chi_square_fit. " *
            "Expected either :terms or :estimate field."
        )
    end
end

"""
    print_fit_result(
        fit
    ) -> Nothing

Print a formatted summary of a least-``\\chi^2`` fit result.

# Function description
This routine is typically called after [`least_chi_square_fit`](@ref) to display
fitted parameters, uncertainties, and fit-quality diagnostics in a compact,
human-readable form.

# Arguments

* `fit`:
  Fit-result object returned by [`least_chi_square_fit`](@ref).

# Returns

`nothing`.

This routine is used for its side effect: it prints a formatted summary to standard output.

# Errors

* No explicit validation is performed.
* Field-access or formatting errors propagate if `fit` does not match the
  expected fit-result layout.

# Notes

* The central extrapolated value is printed as `Result (h→0)`.
* Parameter formatting uses [`AvgErrFormatter.avgerr_e2d_from_float`](@ref)
  when finite values are available, and falls back to explicit scientific
  notation otherwise.
"""
function print_fit_result(
    fit
)
    jobid = nothing

    for i in eachindex(fit.params)
        if !isfinite(fit.params[i]) || !isfinite(fit.param_errors[i])
            tmp_str = @sprintf("%.12e (%.12e)", float(fit.params[i]), float(fit.param_errors[i]))
        else
            tmp_str = AvgErrFormatter.avgerr_e2d_from_float(
                float(fit.params[i]),
                float(fit.param_errors[i])
            )
        end
        JobLoggerTools.println_benji("           λ_$(i-1) = $(tmp_str)", jobid)
    end

    JobLoggerTools.println_benji("", jobid)

    JobLoggerTools.println_benji(
        @sprintf(
            "Chi^2 / d.o.f. = %.12e / %d = %.12e",
            float(fit.chisq),
            fit.dof,
            float(fit.redchisq)
        ),
        jobid
    )

    if !isfinite(fit.estimate) || !isfinite(fit.error_estimate)
        tmp_str = @sprintf(
            "%.12e (%.12e)",
            float(fit.estimate),
            float(fit.error_estimate)
        )
    else
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(
            float(fit.estimate),
            float(fit.error_estimate)
        )
    end

    JobLoggerTools.println_benji("Result (h→0)   = $(tmp_str)", jobid)
    JobLoggerTools.println_benji("", jobid)
end

end  # module LeastChiSquareFit
