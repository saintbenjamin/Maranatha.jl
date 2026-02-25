# ============================================================================
# src/controller/Runner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Runner

using ..JobLoggerTools
using ..Integrate
using ..ErrorEstimator
using ..RichardsonError
using ..FitConvergence

export run_Maranatha

"""
    run_Maranatha(
        integrand,
        a,
        b;
        dim=1,
        nsamples=[4,8,16],
        rule=:simpson13_close,
        err_method::Symbol=:derivative,
        fit_terms::Int=2
    )

Evaluate a definite integral in 1–4 dimensions using a Newton–Cotes quadrature
rule at multiple resolutions, estimate the integration error at each resolution,
and extrapolate the integral to the zero-step limit (`h → 0`) via least-χ² fitting.

# Function description
This function computes a sequence of integral estimates `I(N)` for each `N` in
`nsamples`, along with a corresponding error estimate `err(N)` using the chosen
error method. It then fits a convergence model (rule-dependent) to extrapolate
the integral value as the step size approaches zero.

The convergence fit uses a polynomial-in-`h` basis whose length is controlled by
`fit_terms` (including the constant term). The design matrix used by the fit is:

- column 1: `h^0` (constant term)
- columns 2..fit_terms: `h^(p), h^(p+2), h^(p+4), ...`

where the leading power `p` is determined from `rule`.

# Arguments
- `integrand::Function`: Integrand function. It must accept `dim` positional arguments.
  - `dim == 1` → `integrand(x)`
  - `dim == 2` → `integrand(x, y)`
  - `dim == 3` → `integrand(x, y, z)`
  - `dim == 4` → `integrand(x, y, z, t)`
- `a::Real`: Lower bound (used for every dimension, i.e., the domain is `[a,b]^dim`).
- `b::Real`: Upper bound (used for every dimension, i.e., the domain is `[a,b]^dim`).

# Keyword arguments
- `dim::Int=1`: Dimensionality of the integral (expected range: `1 ≤ dim ≤ 4`).
- `nsamples=[4, 8, 16]`: List of integer resolutions `N` (number of subintervals per axis)
  used to evaluate the integral and error model.
- `rule::Symbol=:simpson13_close`: Quadrature rule symbol forwarded to `integrate_nd`.
- `err_method::Symbol=:derivative`: Error estimation method:
  - `:derivative`  → uses `ErrorEstimator.estimate_error`
  - `:richardson`  → uses `RichardsonError.estimate_error_richardson`
- `fit_terms::Int=2`: Number of basis terms used in the convergence fit (including the
  constant term) forwarded to `fit_convergence` as `nterms`.

# Returns
A 3-tuple:
- `final_estimate::Float64`: Extrapolated integral value (the fitted estimate at `h = 0`).
- `fit_result`: Fit object returned by `fit_convergence` (forwarded unchanged).
- `data::NamedTuple`: Data used for fitting:
  - `data.h::Vector{Float64}`: Step sizes `h = (b-a)/N` for each `N` in `nsamples`.
  - `data.avg::Vector{Float64}`: Raw integral estimates `I(N)`.
  - `data.err::Vector{Float64}`: Error estimates corresponding to each `I(N)`.

# Example
```julia
f(x) = sin(x)
I0, fit, data = run_Maranatha(
    f, 0.0, π;
    dim=1,
    nsamples=[4, 8, 16, 32],
    rule=:simpson13_close,
    fit_terms=4
)
```

# Errors

* Throws an error if err_method is not :derivative or :richardson.
* Any rule-specific constraints on N are enforced by integrate_nd and/or the
chosen error estimator.
"""
function run_Maranatha(
    integrand,
    a,
    b;
    dim=1,
    nsamples=[4,8,16],
    rule=:simpson13_close,
    err_method::Symbol = :derivative,
    fit_terms::Int = 2,
)
    jobid = nothing

    estimates = Float64[]     # List of integral results
    errors = Float64[]        # List of estimated errors
    hs = Float64[]            # List of step sizes

    for N in nsamples
        JobLoggerTools.log_stage_benji("N = $N in $nsamples", jobid)
        h = (b - a) / N
        push!(hs, h)

        # Step 1: Evaluate integral using selected rule
        JobLoggerTools.log_stage_sub1_benji("integrate() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            I = integrate(integrand, a, b, N, dim, rule)
        end
        # Step 2: Estimate integration error
        JobLoggerTools.log_stage_sub1_benji("estimate_error() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            err = if err_method == :derivative
                estimate_error(integrand, a, b, N, dim, rule)
            elseif err_method == :richardson
                estimate_error_richardson(integrand, a, b, N, dim, rule)
            else
                error("Unknown err_method = $err_method (use :derivative or :richardson)")
            end
        end
        push!(estimates, I)
        push!(errors, err)
    end

    # Step 3: Perform least chi-square fit to extrapolate as h → 0
    JobLoggerTools.log_stage_benji("fit_convergence() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        fit_result = fit_convergence(hs, estimates, errors, rule; nterms=fit_terms)
    end
    print_fit_result(fit_result)

    return fit_result.estimate, fit_result, (; h=hs, avg=estimates, err=errors)
end

end  # module Runner