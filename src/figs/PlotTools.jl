# ============================================================================
# src/figs/PlotTools.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PlotTools

using PyPlot

export plot_convergence_result, set_pyplot_latex_style

"""
    set_pyplot_latex_style(
        scale::Float64=0.5
    ) -> Nothing

Set a consistent PyPlot/Matplotlib style using LaTeX (lmodern), similar to the
Deborah.jl plotting style.

# Function description
This function resets Matplotlib rcParams to the defaults and then applies a
LaTeX-enabled style configuration. The `scale` parameter controls overall font
sizes, line widths, and figure size.

# Arguments
- `scale`: Overall scaling factor for font sizes and figure size.

# Returns
- `nothing`.
"""
function set_pyplot_latex_style(
    scale::Float64=0.5
)

    mpl = PyPlot.matplotlib

    rcParams = mpl."rcParams"
    rcParams.update(mpl."rcParamsDefault")

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
    plot_convergence_result(
        name::String, 
        hs::Vector{Float64}, 
        estimates::Vector{Float64},
        errors::Vector{Float64}, 
        fit_result; 
        rule::Symbol=:simpson13_close
    ) -> Nothing

Plot the convergence of a sequence of integral estimates and a fitted convergence
model, and save the figure as a PNG file.

# Function description
This function:
1) Determines the leading convergence power `p` from `rule`.
2) Converts the step sizes `h` into `h^2` for the x-axis.
3) Filters invalid data points (non-finite values, non-positive `h^2`).
4) Builds a convergence model from `fit_result.params` assuming:
   `I(h) = I0 + C1*h^p + C2*h^(p+2) + ...`, where `I0` is taken from
   `fit_result.estimate`.
5) Plots:
   - the fitted curve,
   - the data points with error bars,
   - the extrapolated point at `h^2 = 0` with uncertainty `fit_result.estimate_error`.
6) Saves the plot as `convergence_<name>_<rule>.png`.

This implementation preserves the original plotting logic and file-naming
convention.

# Arguments
- `name`: String label used in the output filename.
- `hs`: Vector of step sizes `h = (b-a)/N`.
- `estimates`: Vector of raw integral estimates corresponding to `hs`.
- `errors`: Vector of error estimates corresponding to `hs` (absolute values are used for plotting).
- `fit_result`: Fit object expected to provide:
  - `fit_result.params`
  - `fit_result.estimate`
  - `fit_result.estimate_error`

# Keyword arguments
- `rule`: Quadrature rule symbol used to select the leading power `p` and to label the output filename.

# Returns
- `nothing`.

# Errors
- Throws an error if input lengths mismatch.
- Throws an error if no valid points remain after filtering.
- Throws an error if `rule` is not recognized.
"""
function plot_convergence_result(
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector{Float64},
    fit_result;
    rule::Symbol = :simpson13_close
)

    # --- Determine leading power from rule ---
    p =
        rule == :simpson13_close ? 4 :
        rule == :simpson13_open  ? 4 :
        rule == :simpson38_close ? 4 :
        rule == :simpson38_open  ? 4 :
        rule == :bode_close      ? 6 :
        rule == :bode_open       ? 6 :
        error("Unknown rule")

    # --- Input checks ---
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        error("Input length mismatch.")
    end

    # Raw x = h^2
    h2 = hs .^ 2
    errors_pos = abs.(errors)

    mask = (h2 .> 0) .& isfinite.(h2) .& isfinite.(estimates) .& isfinite.(errors_pos)

    h2p = h2[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    isempty(h2p) && error("No valid points to plot.")

    # --- New fit result structure ---
    pvec = fit_result.params
    I0      = fit_result.estimate
    I0_err  = fit_result.estimate_error

    # --- Build model automatically from params ---
    # Model: I(h) = I0 + C1*h^p + C2*h^(p+2) + ...
    function model_fit(h)
        s = I0
        for i in 2:length(pvec)
            power = p + 2*(i-2)
            s += pvec[i] * h^power
        end
        return s
    end

    # --- Smooth curve including extrapolated point at h^2 = 0 ---
    h2min = minimum(h2p)
    h2max = maximum(h2p)

    h2_range_log = 10 .^ range(log10(h2min), log10(h2max); length=200)

    # prepend zero explicitly
    h2_range = vcat(0.0, h2_range_log)

    # model needs h, not h^2
    h_range  = sqrt.(h2_range)
    y_fit    = model_fit.(h_range)

    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    # Fit curve
    ax.plot(h2_range, y_fit; color="black", linewidth=2.5)

    # Data points
    ax.errorbar(
        h2p, estp;
        yerr=errp,
        fmt="o",
        color="blue",
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="blue"
    )

    # --- Extrapolated point at h^2 = 0 ---
    ax.errorbar(
        [0.0],
        [I0];
        yerr=[I0_err],
        fmt="s",
        color="red",
        markersize=8,
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="red"
    )

    ax.set_xlabel(raw"$h^2$")
    ax.set_ylabel("Integral Estimate")

    fig.tight_layout()

    outfile = "convergence_$(name)_$(String(rule)).png"
    fig.savefig(outfile)
    PyPlot.close(fig)

    return nothing
end

end  # module PlotTools