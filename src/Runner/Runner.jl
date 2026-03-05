# ============================================================================
# src/Runner/Runner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Runner

using ..Utils.JobLoggerTools
using ..Quadrature
using ..ErrorEstimate
using ..LeastChiSquareFit

export run_Maranatha

"""
    run_Maranatha(
        integrand,
        a,
        b;
        dim::Int = 1,
        nsamples = [4, 8, 12, 16],
        rule::Symbol = :newton_p3,
        boundary::Symbol = :LU_ININ,
        err_method::Symbol = :derivative,
        fit_terms::Int = 2,
        nerr_terms::Int = 1,
        ff_shift::Int = 0,
        use_threads::Bool = false        
    )

High-level execution pipeline for ``n``-dimensional quadrature,
error modeling, and convergence extrapolation.

# Function description
`run_Maranatha` is the orchestration entry point that combines the core
subsystems of `Maranatha.jl`:

- [`Maranatha.Quadrature`](@ref)         : tensor-product quadrature dispatcher (Newton-Cotes / Gauss / B-spline backends)
- [`Maranatha.ErrorEstimate`](@ref)      : residual-based derivative error scale models (midpoint expansion)
- [`Maranatha.LeastChiSquareFit`](@ref)   : least-``\\chi^2`` fitting for ``h \\to 0`` extrapolation

Threaded execution of the derivative-based backend can be enabled with `use_threads`.

For each resolution `N` in `nsamples`, the runner performs:

1. Compute step size ``\\displaystyle{h = \\frac{b-a}{N}}``.

2. Evaluate the integral using the selected rule via [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref).

3. Estimate the integration error according to `err_method`.
   Supported values are `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, and `:enzyme`.
   This dispatches to [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref) and forwards `nerr_terms`,
   allowing optional inclusion of LO, NLO, and higher-order midpoint residual terms in the error model.

4. Accumulate `(h, estimate, error)` triplets.

After processing all resolutions, a weighted convergence fit is performed using a
**residual-informed exponent basis** derived from the rule-family residual model
(dispatched internally based on `rule`) for `(rule, boundary)` and a representative subdivision count.

The fitted convergence model is reconstructed from the stored exponent vector:
```math
I(h) = \\bm{\\lambda}^{\\mathsf{T}} \\varphi(h),
\\qquad
\\varphi_i(h) = h^{\\texttt{powers[i]}} \\quad (i=1,\\dots,n),
```
where `powers = fit_result.powers` with `powers[1] = 0` for the constant term.


Optionally, the fit may apply a **fitting-function-shift** (`ff_shift`) to skip the nominal leading residual power
when the corresponding coefficient is expected to vanish for the given integrand (e.g. symmetry causes the
leading derivative contribution to be zero). In that case, the fitter selects powers starting at
`1 + ff_shift` in the residual-power list and stores the resulting basis in `fit_result.powers`.

The final extrapolated estimate is returned together with the full fit object and the raw data vectors.

# Arguments

* `integrand`:
  Callable integrand. May be a function, closure, or callable struct.
  Must accept `dim` scalar positional arguments.
* `a`, `b`:
  Scalar bounds defining the hypercube domain ``[a,b]^n`` where `n` is the dimensionality.

# Keyword arguments

* `dim::Int = 1`:
  Dimensionality of the tensor-product quadrature.
  Internally dispatched through [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref),
  which supports specialized implementations (from ``1``-dimensional to ``4``-dimensional quadrature)
  and a general ``n``-dimensional quadrature fallback.

* `nsamples = [4, 8, 12, 16]`:
  Vector of subdivision counts `N`. Each value defines a different grid resolution used in the convergence study.

* `rule::Symbol = :newton_p3`:
  Quadrature rule identifier forwarded to integration, error estimation, and fitting.
  Examples include Newton-Cotes NS rules (`:newton_pK`) as well as Gauss-family and B-spline rule symbols
  supported by [`Maranatha.Quadrature`](@ref).

* `boundary::Symbol = :LU_ININ`:
  Boundary pattern for the composite rule assembly. Supported values are:
  `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`.
  This is forwarded consistently to integration, error estimation, and fitting.

* `err_method`:
  Backend used for derivative evaluation via [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative`](@ref).
  Supported values: `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, `:enzyme`.
  (dispatching to [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)).
  This keyword is reserved for future error-estimation backends.

* `fit_terms::Int = 2`:
  Number of basis terms used in the convergence model (including the constant term).
  This is forwarded as `nterms` to the least-``\\chi^2`` fitter.
  The fitter selects `nterms-1` residual-derived exponents (optionally shifted by `ff_shift`)
  and stores the full exponent vector (with leading `0`) in `fit_result.powers`.

* `nerr_terms::Int = 1`:
  Number of midpoint residual terms used by the derivative-based error estimator.
  This is forwarded to [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref) as `nerr_terms`.

  * ``1``  uses LO only
  * ``>1`` uses LO + NLO + ... up to `nerr_terms` terms (subject to the residual scan limit)

* `ff_shift::Int = 0`:
  Forward shift applied inside the fitter when selecting residual-derived powers.
  If `ff_shift = 1`, the fitter skips the first residual power and fits using the next ones, etc.
  This is forwarded to [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref) as `ff_shift`.

* `use_threads::Bool = false`:
  If `true`, dispatches to the threaded error-estimation backend ([`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_threads`](@ref)).

# Returns

A 3-tuple:

* `final_estimate::Float64`:
  Convenience alias for `fit_result.estimate`.

* `fit_result::NamedTuple`:
  Fit object returned by [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
  Key fields:

  * `estimate::Float64`:
    Extrapolated integral estimate ``I_0`` (the ``h \\to 0`` limit), equal to `params[1]`.
  * `error_estimate::Float64`:
    `1\\sigma` uncertainty of `estimate`, taken from `param_errors[1]`.
  * `params::Vector{Float64}`:
    Fitted parameter vector ``[I_0, C_1, C_2, \\ldots]``.
  * `param_errors::Vector{Float64}`:
    `1\\sigma` uncertainties for `params` (square roots of `diag(cov)`).
  * `cov::Matrix{Float64}`:
    Parameter covariance matrix, suitable for uncertainty propagation
    (e.g. ``\\sigma_{\\mathrm{fit}}(h)^2 = \\varphi(h)^{\\top} \\, V \\, \\varphi(h)``).
  * `powers::Vector{Int}`:
    Exponent vector used in the fit basis (with `powers[1] = 0`), reflecting any `ff_shift` applied.
  * `chisq::Float64`, `redchisq::Float64`, `dof::Int`:
    Standard ``\\chi^2`` diagnostics.

* `data::NamedTuple`:
  Raw convergence data:

  * `h`   : step sizes
  * `avg` : integral estimates
  * `err` : error estimates

# Design notes

* The runner is **dimension-agnostic** and supports arbitrary ``n \\ge 1`` subject to computational cost.
* Error estimators provide a *scale model* rather than a strict truncation bound, enabling stable weighted fits.
* Logging and timing are centralized through [`Maranatha.Utils.JobLoggerTools`](@ref).
* Threaded error estimation is optionally enabled via `use_threads`, without affecting the fitting stage.

# Example

```julia
f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

I0, fit, data = run_Maranatha(
    f,
    0.0, 1.0;
    dim=4,
    nsamples=[40, 44, 48, 52, 56, 60, 64],
    rule=:newton_p5,
    boundary=:LU_ININ,
    err_method=:derivative,
    fit_terms=4,
    nerr_terms=2,
    ff_shift=1
)
```

"""
function run_Maranatha(
    integrand,
    a,
    b;
    dim=1,
    nsamples=[4,8,12,16],
    rule=:newton_p3,
    boundary=:LU_ININ,
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    fit_terms::Int = 2,
    nerr_terms::Int = 1,
    ff_shift::Int = 0,
    use_threads::Bool = false,
)
    jobid = nothing

    # ------------------------------------------------------------
    # Normalize legacy inputs (do this ONCE, early)
    # ------------------------------------------------------------
    rule = normalize_newton_rule(rule)
    rule = normalize_bspline_rule(rule)
    boundary = normalize_boundary(boundary)

    estimates = Float64[]     # List of integral results
    errors = Float64[]        # List of estimated errors
    hs = Float64[]            # List of step sizes

    for N in nsamples
        # JobLoggerTools.log_stage_benji("N = $N in $nsamples", jobid)
        h = (b - a) / N
        push!(hs, h)

        # # Step 1: Evaluate integral using selected rule
        # JobLoggerTools.log_stage_sub1_benji("quadrature() ::", jobid)
        # JobLoggerTools.@logtime_benji jobid begin
        I = Quadrature.QuadratureDispatch.quadrature(integrand, a, b, N, dim, rule, boundary)
        # end
        # # Step 2: Estimate integration error
        # JobLoggerTools.log_stage_sub1_benji("error_estimate() ::", jobid)
        # JobLoggerTools.@logtime_benji jobid begin
        err = if use_threads
            ErrorEstimate.ErrorDispatch.error_estimate_threads(integrand, a, b, N, dim, rule, boundary; err_method=err_method, nerr_terms=nerr_terms)
        else
            ErrorEstimate.ErrorDispatch.error_estimate(integrand, a, b, N, dim, rule, boundary; err_method=err_method, nerr_terms=nerr_terms)
        end
        # end
        push!(estimates, I)
        push!(errors, err)
    end

    # Step 3: Perform least chi-square fit to extrapolate as h → 0
    # JobLoggerTools.log_stage_benji("least_chi_square_fit() ::", jobid)
    # JobLoggerTools.@logtime_benji jobid begin
        fit_result = LeastChiSquareFit.least_chi_square_fit(a, b, hs, estimates, errors, rule, boundary; nterms=fit_terms, ff_shift=ff_shift)
    # end
    LeastChiSquareFit.print_fit_result(fit_result)

    return fit_result.estimate, fit_result, (; h=hs, avg=estimates, err=errors)
end

"""
    _DEPRECATION_WARNED :: Set{Symbol}

Session-local registry of symbols that have already triggered a deprecation warning.

# Description
This set is used internally by Maranatha's normalization helpers
(e.g. rule and boundary normalizers) to ensure that each deprecated
symbol emits a warning **only once per Julia session**.

When a legacy symbol is first encountered, it is inserted into this set
after the warning is issued. Subsequent encounters of the same symbol
are silently ignored to prevent repeated warning spam.

# Notes
- Intended for internal use only.
- The registry is reset when the Julia session restarts.
"""
const _DEPRECATION_WARNED = Set{Symbol}()

"""
    _warn_once(
        sym::Symbol, 
        msg::String
    ) -> Nothing

Emit a deprecation warning only once per session for a given symbol.

# Function description
This helper is used internally by rule and boundary normalizers to avoid
repeated warning spam when legacy symbols are encountered multiple times
during a single Julia session.

If `sym` has not yet triggered a warning, `msg` is emitted via `@warn`
and the symbol is recorded. Subsequent calls with the same `sym` are ignored.

# Arguments
- `sym`: The legacy symbol that triggered the warning.
- `msg`: Warning message to be emitted.

# Notes
- This function is intended for internal use.
- The warning registry is session-local.
"""
@inline function _warn_once(
    sym::Symbol, 
    msg::String
)::Nothing
    if !(sym in _DEPRECATION_WARNED)
        push!(_DEPRECATION_WARNED, sym)
        @warn msg
    end
    return nothing
end

"""
    normalize_newton_rule(
        rule::Symbol; 
        warn::Bool = true
    ) -> Symbol

Normalize legacy Newton-Cotes rule symbols to the current naming convention.

# Function description
This function converts deprecated Newton rule symbols of the form

- `:ns_pK`

into the current canonical format

- `:newton_pK`

where `K` denotes the polynomial degree.

If `rule` is already in the new format or does not belong to the
legacy Newton family, it is returned unchanged.

# Arguments
- `rule`: Quadrature rule symbol.
- `warn`: If `true`, emit a deprecation warning (once per symbol).

# Returns
- `Symbol`: The normalized rule symbol.

# Notes
- This function should be called early in high-level runners
  (e.g. `run_Maranatha`) to ensure internal consistency.
"""
function normalize_newton_rule(
    rule::Symbol; 
    warn::Bool = true
)::Symbol
    s = String(rule)

    if startswith(s, "ns_p")
        # preserve suffix verbatim (supports multi-digit like ns_p10)
        suffix = s[length("ns_p")+1:end]  # e.g. "3", "10"
        new_rule = Symbol("newton_p", suffix)

        if warn
            _warn_once(rule,
                "Deprecated rule symbol `$rule` detected. " *
                "Use `$new_rule` instead. It will be normalized automatically for now."
            )
        end
        return new_rule
    end

    return rule
end

"""
    normalize_bspline_rule(
        rule::Symbol; 
        warn::Bool = true
    ) -> Symbol

Normalize legacy B-spline rule symbols to the current naming convention.

# Function description
This function converts deprecated B-spline rule symbols:

- `:bsplI_pK` → `:bspline_interp_pK`
- `:bsplS_pK` → `:bspline_smooth_pK`

where `K` is the spline degree.

If `rule` is already in the new format or does not belong to the
legacy B-spline family, it is returned unchanged.

# Arguments
- `rule`: Quadrature rule symbol.
- `warn`: If `true`, emit a deprecation warning (once per symbol).

# Returns
- `Symbol`: The normalized rule symbol.

# Notes
- This function only renames the rule; it does not parse the spline degree.
  Use `_parse_bspl_p` for degree extraction.
"""
function normalize_bspline_rule(
    rule::Symbol; 
    warn::Bool = true
)::Symbol
    s = String(rule)

    if startswith(s, "bsplI_p")
        deg = s[length("bsplI_p")+1:end]          # after "bsplI_p"
        new_rule = Symbol("bspline_interp_p", deg)

        if warn
            _warn_once(rule,
                "Deprecated B-spline rule symbol `$rule` detected. " *
                "Use `$new_rule` instead. It will be normalized automatically for now."
            )
        end
        return new_rule

    elseif startswith(s, "bsplS_p")
        deg = s[length("bsplS_p")+1:end]          # after "bsplS_p"
        new_rule = Symbol("bspline_smooth_p", deg)

        if warn
            _warn_once(rule,
                "Deprecated B-spline rule symbol `$rule` detected. " *
                "Use `$new_rule` instead. It will be normalized automatically for now."
            )
        end
        return new_rule
    end

    return rule
end

"""
    normalize_boundary(
        boundary::Symbol; 
        warn::Bool = true
    ) -> Symbol

Normalize legacy boundary symbols to the current LU-based naming scheme.

# Function description
This function converts deprecated boundary symbols:

- `:LCRC` → `:LU_ININ`
- `:LORC` → `:LU_EXIN`
- `:LCRO` → `:LU_INEX`
- `:LORO` → `:LU_EXEX`

where:

- `IN` = include endpoint (closed)
- `EX` = exclude endpoint (opened)

If `boundary` already follows the `:LU_*` convention, it is returned unchanged.

# Arguments
- `boundary`: Boundary configuration symbol.
- `warn`: If `true`, emit a deprecation warning (once per symbol).

# Returns
- `Symbol`: The normalized boundary symbol.

# Notes
- Normalization should occur before boundary decoding.
- This function does not validate semantic consistency beyond renaming.
"""
function normalize_boundary(
    boundary::Symbol; 
    warn::Bool = true
)::Symbol
    s = String(boundary)

    # already new format
    if startswith(s, "LU_")
        return boundary
    end

    new_boundary = if boundary === :LCRC
        :LU_ININ
    elseif boundary === :LORC
        :LU_EXIN
    elseif boundary === :LCRO
        :LU_INEX
    elseif boundary === :LORO
        :LU_EXEX
    else
        boundary
    end

    if warn && (new_boundary !== boundary)
        _warn_once(boundary,
            "Deprecated boundary symbol `$boundary` detected. " *
            "Use `$new_boundary` instead. It will be normalized automatically for now."
        )
    end

    return new_boundary
end

end  # module Runner