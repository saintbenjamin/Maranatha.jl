module FitConvergence

using LsqFit
using Statistics

export fit_convergence

"""
    fit_convergence(hs::Vector{<:Real}, estimates::Vector{<:Real}, errors::Vector{<:Real},
                    rule::Symbol; dim::Int=1) -> NamedTuple

Perform a weighted least-squares fit to extrapolate the integral value  
as step size `h → 0`, using the known error structure of the specified integration rule.

# Arguments
- `hs`: Vector of step sizes `h = (b - a) / N`
- `estimates`: Vector of raw integral approximations at each `h`
- `errors`: Estimated integration errors (used as weights in the fit)
- `rule`: Integration rule used (`:simpson13`, `:simpson38`, or `:bode`)
- `dim`: Dimension of the integral (default = 1); affects the error model

# Returns
- `NamedTuple`:
    - `estimate`: Extrapolated value as `h → 0` (i.e., `fit.param[1]`)
    - `fit`: Full `LsqFitResult` object from `LsqFit.curve_fit`

# Notes
- For 1D integrals (`dim=1`), fits to:  
      `I(h) ≈ I₀ + C₁·h^p + C₂·h^{p+2} + C₃·h^{p+4}`  
  to capture higher-order corrections.
- For higher dimensions (`dim>1`), uses simpler model:  
      `I(h) ≈ I₀ + C·h^p`
- Uses robust error fallback to prevent division-by-zero if any `errors[i] ≤ 0`.
"""
function fit_convergence(hs, estimates, errors, rule::Symbol; dim::Int=1)
    # Determine leading error power
    p = rule == :simpson13 ? 4 :
        rule == :simpson38 ? 4 :
        rule == :bode ? 6 :
        error("Unknown rule")

    # Define error model by dimension
    if dim == 1
        model_1d(h, pars) = pars[1] .+ pars[2]*h.^p .+ pars[3]*h.^(p+2) .+ pars[4]*h.^(p+4)
        model = model_1d
        init_params = [mean(estimates), 0.0, 0.0, 0.0]
    else
        model_nd(h, pars) = pars[1] .+ pars[2]*h.^p
        model = model_nd
        init_params = [mean(estimates), 0.0]
    end

    # Safe error weights
    safe_errors = map(e -> e > 0 ? e : 1e-8, errors)

    fit = curve_fit(model, hs, estimates, safe_errors, init_params)

    return (; estimate = fit.param[1], fit = fit)
end

end # module