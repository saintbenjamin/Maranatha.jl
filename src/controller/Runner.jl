module Runner

using ..Integrate
using ..ErrorEstimator
using ..FitConvergence

export run_Maranatha

"""
    run_Maranatha(integrand, a, b; dim=1, nsamples=[4,8,16], rule=:simpson13_close)

Evaluate a definite integral in 1–4 dimensions using Newton–Cotes quadrature  
rules and extrapolate to zero step size via least χ² fitting.

# Arguments
- `integrand::Function`: Function of 1 to 4 variables depending on `dim`.
- `a::Real`, `b::Real`: Integration bounds (applied identically across all dimensions).
- `dim::Int`: Dimensionality of the integral (1 ≤ dim ≤ 4).
- `nsamples::Vector{Int}`: List of sample sizes `N` to use, e.g., `[4, 8, 16]`.
- `rule::Symbol`: Integration rule; one of `:simpson13_close`, `:simpson38_close`, or `:bode_close`.

# Returns
A 3-tuple:
- `final_estimate::Float64`: Extrapolated integral value as `h → 0`.
- `fit_result::LsqFit.LsqFitResult`: Full result of the χ² fitting.
- `NamedTuple`: Data used in fitting: `(h, avg, err)` where
    - `h`: Vector of step sizes,
    - `avg`: Vector of raw integral estimates,
    - `err`: Vector of estimated integration errors.

# Example
```julia
f(x) = sin(x)
I, fit, data = run_Maranatha(f, 0.0, π; dim=1, nsamples=[4, 8, 16, 32], rule=:simpson13_close)
```
"""
function run_Maranatha(integrand, a, b; dim=1, nsamples=[4,8,16], rule=:simpson13_close)

    estimates = Float64[]     # List of integral results
    errors = Float64[]        # List of estimated errors
    hs = Float64[]            # List of step sizes

    for N in nsamples
        h = (b - a) / N
        push!(hs, h)

        # Step 1: Evaluate integral using selected rule
        I = integrate_nd(integrand, a, b, N, dim, rule)

        # Step 2: Estimate integration error
        err = estimate_error(integrand, a, b, N, dim, rule)

        push!(estimates, I)
        push!(errors, err)
    end

    # Step 3: Perform least chi-square fit to extrapolate as h → 0
    fit_result = fit_convergence(hs, estimates, errors, rule; dim=dim)
    print_fit_result(fit_result)

    return fit_result.estimate, fit_result, (; h=hs, avg=estimates, err=errors)
end

end