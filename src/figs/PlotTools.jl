module PlotTools

using Plots

export plot_convergence_result

"""
    plot_convergence_result(name::String, hs::Vector{Float64}, estimates::Vector{Float64},
                            errors::Vector{Float64}, fit_result; rule::Symbol=:simpson13)

Generate and save a convergence plot for numerical integration results.  
This plot visualizes how integral estimates approach the extrapolated value  
as the step size decreases, with error bars and a least-squares fit curve.

# Arguments
- `name`: Identifier for the output filename (used as prefix)
- `hs`: Vector of step sizes `h`
- `estimates`: Vector of integral estimates at each `h`
- `errors`: Vector of error estimates (used for vertical error bars)
- `fit_result`: Output from `fit_convergence`, including `estimate` and fit parameters
- `rule`: Symbol for the integration rule used (`:simpson13`, `:simpson38`, `:bode`)

# Output
- Saves plot as a PNG file: `"convergence_$(name)_$(rule).png"`

# Plot Features
- X-axis: Step size squared (`h²`), displayed on a **logarithmic scale**
- Y-axis: Integral estimates, linear scale
- Includes:
    - Blue scatter points with transparent error bars
    - Extrapolation curve from least-squares fit
    - LaTeX-formatted axis labels using Computer Modern font

# Notes
- Fit function follows the model `I(h) ≈ I₀ + C₁·h^p + C₂·h^{p+2} + ...`
- Automatically adapts the curve degree based on `fit_result.fit.param`
"""
function plot_convergence_result(name::String, hs::Vector{Float64}, estimates::Vector{Float64},
                                 errors::Vector{Float64}, fit_result; rule::Symbol=:simpson13)

    h2 = hs .^ 2
    pvec = fit_result.fit.param
    I₀ = pvec[1]

    # Define full model function using all available parameters
    model_fit = h -> begin
        s = I₀
        for (i, c) in enumerate(pvec[2:end])
            s += c * h^(2*(i))  # powers: h^p, h^{p+2}, h^{p+4}, ...
        end
        return s
    end

    h2_range = range(minimum(h2), maximum(h2); length=200)
    h_range = sqrt.(h2_range)

    sf = 2
    plot_font = "Computer Modern"
    Plots.default(
        fontfamily=plot_font,
        thickness_scaling=sf,
        linewidth=1/sf,
        tickfontsize=12/sf,
        legendfontsize=12/sf,
        guidefontsize=16/sf,
        dpi=500,
        formatter=:latex,
        tex_output_standalone=true
    )

    plt = plot(
        h2, estimates, yerror = errors,
        label = "Numerical Estimates ± Error",
        xlabel = "\$h^2\$ (Step size squared)", 
        ylabel = "Integral Estimate",
        xscale = :log10,
        # title = "Convergence of Integration Result ($(Symbol(rule)))",
        legend = :none, 
        seriestype = :scatter,
        lw = 2,
        markercolor       = Plots.RGBA(0.0, 0.0, 1.0, 0.5),
        markerstrokecolor = Plots.RGBA(0.0, 0.0, 1.0, 0.5),
        markerstrokewidth = 1.0,
        size = (800, 600)
    )

    plot!(plt, h2_range, model_fit.(h_range), lw=4,
          label = "Least-Squares Fit")

    savefig(plt, "convergence_$(name)_$(String(rule)).png")
end

end