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
    plot_convergence_result(name, hs, estimates, errors, fit_result; rule=:simpson13_close)

Plot convergence using weighted linear χ² fit result.
Compatible with new FitConvergence module (no LsqFit dependency).
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
        rule == :simpson13_close            ? 4 :
        rule == :simpson13_open  ? 4 :
        rule == :simpson38_close            ? 4 :
        rule == :simpson38_open  ? 4 :
        rule == :bode_close                 ? 6 :
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
    # I0   = pvec[1]
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

    # Smooth curve
    h2min = minimum(h2p)
    h2max = maximum(h2p)

    h2_range = 10 .^ range(log10(h2min), log10(h2max); length=200)
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

end