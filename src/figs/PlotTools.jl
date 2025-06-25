module PlotTools

using Plots

export plot_convergence_result

"""
    plot_convergence_result(name::String, hs::Vector{Float64}, estimates::Vector{Float64},
                            errors::Vector{Float64}, fit_result; rule::Symbol=:simpson13)

Generate and save a convergence plot for numerical integration results.  
This plot visualizes how integral estimates approach the extrapolated value  
as the step size decreases, and overlays a least-squares fit curve.

# Arguments
- `name`: Identifier for output filename (used as prefix)
- `hs`: Vector of step sizes (h)
- `estimates`: Vector of raw integral values at each h
- `errors`: Error bars (used as vertical uncertainty)
- `fit_result`: Output from `fit_convergence`, containing extrapolated I₀ and slope
- `rule`: Integration rule used (`:simpson13`, `:simpson38`, `:bode`); used in title/filename

# Output
- Saves plot as PNG file: `"convergence_\$(name)_\$(rule).png"`

# Plot Details
- X-axis: `h²` (log scale)  
- Y-axis: Estimated integral (log scale)  
- Includes:  
    - Scatter with error bars  
    - Least-squares fit curve `I(h²) ≈ I₀ + C * h²`
"""
function plot_convergence_result(name::String, hs::Vector{Float64}, estimates::Vector{Float64},
                                 errors::Vector{Float64}, fit_result; rule::Symbol=:simpson13)

    h2 = hs .^ 2
    I₀, C = fit_result.estimate, fit_result.fit.param[2]
    model_fit = h2 -> I₀ .+ C * h2
    h2_range = range(minimum(h2), maximum(h2); length=100)

    plt = scatter(h2, estimates;
        yerror = errors,
        xscale = :log10, yscale = :log10,
        label = "Numerical Estimates ± Error",
        xlabel = "h² (Step size squared)", ylabel = "Integral Estimate",
        title = "Convergence of Integration Result ($(Symbol(rule)))",
        legend = :bottomright, markersize = 6)

    plot!(plt, h2_range, model_fit.(h2_range),
          label = "Least-Squares Fit", lw = 2)

    savefig(plt, "convergence_$(name)_$(String(rule)).png")
end

end