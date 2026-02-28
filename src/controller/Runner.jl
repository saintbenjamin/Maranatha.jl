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
using ..LeastChiSquareFit

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

High-level execution pipeline for ``n``-dimensional quadrature,
error modeling, and convergence extrapolation.

# Function description
`run_Maranatha` is the orchestration entry point that combines the core
subsystems of Maranatha:

- [`Maranatha.Integrate`](@ref)      : tensor-product Newton-Cotes quadrature in **arbitrary** dimension
- [`Maranatha.ErrorEstimator`](@ref) : derivative-based error scale models
- [`Maranatha.LeastChiSquareFit`](@ref) : least-χ² fitting for ``h \\to 0`` extrapolation

For each resolution `N` in `nsamples`, the runner performs:

1. Compute step size
   ```math
   h = \\frac{b-a}{N} \\, .
   ```

2. Evaluate the integral using the selected rule via [`Maranatha.Integrate.integrate`](@ref).

3. Estimate the integration error according to `err_method`.

4. Accumulate `(h, estimate, error)` triplets.

After processing all resolutions, a weighted convergence fit is performed:
```math
I(h) = I_0 + C_1 \\, h^p + C_2 \\, h^{p+2} + ...
```
where the leading power ``p`` is determined from `rule`.

The final extrapolated estimate ``I_0`` is returned together with the full
fit object and the raw data vectors used in the fit.

# Arguments
- `integrand`: Callable integrand. May be a function, closure, or callable struct.
  Must accept `dim` scalar positional arguments.
- `a`, `b`: Scalar bounds defining the hypercube domain ``[a,b]^n`` where ``n`` is (spacetime) dimensionality.

# Keyword arguments
- `dim::Int=1`:
  Dimensionality of the tensor-product quadrature.
  Internally dispatched through [`Maranatha.Integrate.integrate`](@ref), 
  which supports specialized implementations (from ``1``-dimensional to ``4``-dimensional quadrature) and a general ``n``-dimensional quadrature fallback.

- `nsamples=[4,8,16]`:
  Vector of subdivision counts `N`. Each value defines a different grid
  resolution used in the convergence study.

- `rule::Symbol=:simpson13_close`:
  Newton-Cotes rule identifier forwarded to both integration and error
  estimation modules.

- `err_method::Symbol=:derivative`:
  Error estimation strategy.
  Supported values:

  - `:derivative`  → [`Maranatha.ErrorEstimator.estimate_error`](@ref)
  
- `fit_terms::Int=2`:
  Number of basis terms used in the convergence model
  (including the constant extrapolated value).

# Returns
A 3-tuple:

- `final_estimate::Float64`:
  Convenience alias for `fit_result.estimate`.

- `fit_result::NamedTuple`:
  Fit object returned by [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).  
  Fields:
  - `estimate::Float64` :
    Extrapolated integral estimate ``I_0`` (the ``h \\to 0`` limit), equal to `params[1]`.
  - `estimate_error::Float64` :
    ``1 \\, \\sigma`` uncertainty of `estimate`, taken as `param_errors[1]` from the covariance diagonal.
  - `params::Vector{Float64}` :
    Fitted parameter vector ``[I_0, C_1, C_2, \\ldots]`` for the model
    ``I(h) = I_0 + C_1 h^p + C_2 h^{p+2} + \\cdots``.
  - `param_errors::Vector{Float64}` :
    ``1 \\, \\sigma`` uncertainties for `params` (square roots of `diag(cov)`).
  - `cov::Matrix{Float64}` :
    Parameter covariance matrix, suitable for uncertainty propagation
    (e.g. ``\\sigma_{\\mathrm{fit}}(h)^2 = \\phi(h)^{\\top}\\,V\\,\\phi(h)`` where ``V`` is covariance matrix).
  - `chisq::Float64` :
    Total chi-square value 
    ```math
    \\chi^2 = \\sum_i \\left(\\frac{y_i - \\hat y_i}{\\sigma_i}\\right)^2 \\,.
    ```
  - `redchisq::Float64` :
    Reduced chi-square ``\\chi^2/\\mathrm{d.o.f.}``.
    (If `dof == 0`, this may be `Inf`/`NaN` depending on `chisq`.)
  - `dof::Int` :
    Degrees of freedom `length(estimates) - length(params)`.

- `data::NamedTuple`:
  Raw convergence data:
  - `h`   : step sizes
  - `avg` : integral estimates
  - `err` : error estimates

# Design notes
- The runner is **dimension-agnostic**: the tensor-product implementation
  allows arbitrary ``n \\ge 1`` (``n``: dimension), subject only to computational cost.
- Error estimators provide a *scale model* rather than a strict truncation bound,
  enabling stable weighted fits across dimensions.
- Logging and timing are fully centralized through [`Maranatha.JobLoggerTools`](@ref).

# Example
```julia
f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

I0, fit, data = run_Maranatha(
    f, 
    0.0, 1.0;
    dim=4,
    nsamples=[40, 44, 48, 52, 56, 60, 64],
    rule=:bode_close,
    err_method=:derivative
    fit_terms=4
)
```

"""
function run_Maranatha(
    integrand,
    a,
    b;
    dim=1,
    nsamples=[4,8,16],
    rule=:ns_p3,
    boundary=:LCRC,
    err_method::Symbol = :derivative,
    fit_terms::Int = 2,
)
    jobid = nothing

    estimates = Float64[]     # List of integral results
    errors = Float64[]        # List of estimated errors
    hs = Float64[]            # List of step sizes

    for N in nsamples
        # JobLoggerTools.log_stage_benji("N = $N in $nsamples", jobid)
        h = (b - a) / N
        push!(hs, h)

        # # Step 1: Evaluate integral using selected rule
        # JobLoggerTools.log_stage_sub1_benji("integrate() ::", jobid)
        # JobLoggerTools.@logtime_benji jobid begin
            I = integrate(integrand, a, b, N, dim, rule, boundary)
        # end
        # # Step 2: Estimate integration error
        # JobLoggerTools.log_stage_sub1_benji("estimate_error() ::", jobid)
        # JobLoggerTools.@logtime_benji jobid begin
            err = if err_method == :derivative
                estimate_error(integrand, a, b, N, dim, rule, boundary)
            else
                JobLoggerTools.error_benji("Unknown err_method = $err_method (use :derivative)")
            end
        # end
        push!(estimates, I)
        push!(errors, err)
    end

    # Step 3: Perform least chi-square fit to extrapolate as h → 0
    # JobLoggerTools.log_stage_benji("least_chi_square_fit() ::", jobid)
    # JobLoggerTools.@logtime_benji jobid begin
        fit_result = least_chi_square_fit(a, b, hs, estimates, errors, rule, boundary; nterms=fit_terms)
    # end
    print_fit_result(fit_result)

    return fit_result.estimate, fit_result, (; h=hs, avg=estimates, err=errors)
end

end  # module Runner