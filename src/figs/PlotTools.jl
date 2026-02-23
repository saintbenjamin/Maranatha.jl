module PlotTools

using PyPlot

export plot_convergence_result, set_pyplot_latex_style

"""
    set_pyplot_latex_style(scale::Float64=0.5)

Set a consistent PyPlot/Matplotlib style using LaTeX (lmodern),
similar to the Deborah.jl plotting style.

- `scale`: overall scaling factor for font sizes and figure size.
"""
function set_pyplot_latex_style(scale::Float64=0.5)
    rcParams = PyPlot.matplotlib["rcParams"]
    rcParams.update(PyPlot.matplotlib["rcParamsDefault"])
    rcParams.update(Dict(
        "figure.figsize" => (16 * scale, 12 * scale),
        "font.size" => 24 * scale,
        "axes.labelsize" => 24 * scale,
        "legend.fontsize" => 24 * scale,
        "lines.markersize" => 18 * scale,
        "lines.linewidth" => 4 * scale,
        "font.family" => "lmodern",
        "text.usetex" => true,
        "text.latex.preamble" => raw"\usepackage{lmodern}",
        "axes.grid" => true,
        "grid.alpha" => 0.3
    ))
    return nothing
end

"""
    plot_convergence_result(name::String, hs::Vector{Float64}, estimates::Vector{Float64},
                            errors::Vector{Float64}, fit_result; rule::Symbol=:simpson13)

Generate and save a convergence plot for numerical integration results
using PyPlot.jl (Matplotlib backend).
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
- Saves plot as a PNG file: `"convergence_\$(name)_\$(rule).png"`

# Plot Features
- X-axis: `h^2` in log scale
- Y-axis: integral estimates (linear)
- Scatter + vertical error bars
- Least-squares fit curve from `fit_result.fit.param`

# Robustness
- Sanitizes `errors` to be non-negative (uses `abs`)
- Filters out invalid points (NaN/Inf, and non-positive `h^2` for log scale)
- Prints a warning if any points were dropped

# Notes
- Fit function follows the model `I(h) ≈ I₀ + C₁·h^p + C₂·h^{p+2} + ...`
- Automatically adapts the curve degree based on `fit_result.fit.param`
"""
function plot_convergence_result(
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector{Float64},
    fit_result;
    rule::Symbol = :simpson13
)
    # Basic shape checks (keep it simple but explicit)
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        error("Input length mismatch: hs=$(n), estimates=$(length(estimates)), errors=$(length(errors)).")
    end

    # Raw x = h^2
    h2 = hs .^ 2

    # Matplotlib requires yerr >= 0 and finite.
    # Use abs to remove sign; this is the most conservative interpretation:
    # error bars represent magnitudes.
    errors_pos = abs.(errors)

    # Filter mask for log scale and finiteness
    mask = (h2 .> 0) .& isfinite.(h2) .& isfinite.(estimates) .& isfinite.(errors_pos)

    ndrop = n - count(mask)
    if ndrop > 0
        @warn "plot_convergence_result: dropped $(ndrop) invalid point(s) (needs h^2>0 and all finite; yerr made non-negative by abs)."
    end

    h2p = h2[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    if isempty(h2p)
        error("No valid points left to plot after filtering (need h^2>0 and finite values).")
    end

    # Fit params: I(h) ≈ I0 + c1*h^2 + c2*h^4 + ...
    pvec = fit_result.fit.param
    I₀ = pvec[1]

    model_fit(h) = begin
        s = I₀
        for (i, c) in enumerate(pvec[2:end])
            s += c * h^(2*i)
        end
        return s
    end

    # Smooth curve range in h^2 (log-spaced, using filtered range)
    h2min = minimum(h2p)
    h2max = maximum(h2p)
    if h2min <= 0
        error("Internal error: h2min <= 0 after filtering; cannot build log-spaced curve.")
    end
    h2_range = 10 .^ range(log10(h2min), log10(h2max); length=200)
    h_range = sqrt.(h2_range)
    y_fit = model_fit.(h_range)

    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6, 5.0), dpi=500)

    # Fit curve
    ax.plot(h2_range, y_fit; color = "red", linewidth=2.5)

    # Error bars + scatter
    ax.errorbar(
        h2p, estp;
        yerr = errp,
        fmt = "o",
        color = "blue",
        alpha = 1.0,
        capsize = 6,
        markerfacecolor="none", 
        markeredgecolor="blue"

    )

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel(raw"$h^2$ (Step size squared)")
    ax.set_ylabel("Integral Estimate")

    fig.tight_layout()

    # Save (keep filename convention)
    outfile = "convergence_$(name)_$(String(rule)).png"
    fig.savefig(outfile)
    PyPlot.close(fig)

    return nothing
end

end