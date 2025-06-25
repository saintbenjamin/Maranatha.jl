module FitConvergence

using LsqFit
using Statistics

export fit_convergence

"""
    fit_convergence(hs::Vector{<:Real}, estimates::Vector{<:Real}, errors::Vector{<:Real}, rule::Symbol) -> NamedTuple

Perform a weighted least-squares fit to extrapolate the integral value  
as step size `h → 0`, using the known leading-order error of the integration rule.

# Arguments
- `hs`: Vector of step sizes `h = (b - a) / N`
- `estimates`: Vector of raw integral approximations at each `h`
- `errors`: Estimated integration errors (used as weights in the fit)
- `rule`: Integration rule used (`:simpson13`, `:simpson38`, or `:bode`); determines convergence order

# Returns
- `NamedTuple`:  
    - `estimate`: Extrapolated value as `h → 0` (i.e., fit.param[1])  
    - `fit`: Full `LsqFitResult` object from `LsqFit.curve_fit`  

# Notes
- Model: `I(h) ≈ I₀ + C * h^p`, where `p` is 4 or 6 depending on the rule  
- Automatically avoids division by zero if `errors` contain non-positive values
"""
function fit_convergence(hs, estimates, errors, rule)
    # Determine leading error power based on rule
    p = rule == :simpson13 ? 4 : rule == :simpson38 ? 4 : rule == :bode ? 6 : error("Unknown rule")

    model(h, pars) = pars[1] .+ pars[2] .* h .^ p
    init_params = [mean(estimates), 0.0]

    # Replace non-positive errors with small ε to avoid divide-by-zero
    safe_errors = map(e -> e > 0 ? e : 1e-8, errors)

    fit = curve_fit(model, hs, estimates, safe_errors, init_params)

    return (; estimate = fit.param[1], fit = fit)
end

end # module