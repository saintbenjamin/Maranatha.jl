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
using LinearAlgebra
using ..JobLoggerTools

export plot_convergence_result, set_pyplot_latex_style

"""
    set_pyplot_latex_style(
        scale::Float64=0.5
    ) -> Nothing

Configure [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering and appropriate font settings for publications.

This function modifies [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) to enable [``\\LaTeX``](https://www.latex-project.org/)-based text rendering and adjust 
font sizes, marker sizes, and line widths for consistent visual output.  
Useful for generating high-quality plots for papers or presentations.

# Arguments
- `scale::Float64`: Scaling factor for font sizes and figure dimensions. Default is `0.5`.

# Side Effects
- Modifies [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)'s global rendering configuration via [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams).
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
        rule::Symbol = :simpson13_close
    ) -> Nothing

Plot convergence data ``I(h)`` against ``h^2``, overlay the fitted extrapolation curve,
and visualize the **fit uncertainty band** propagated from the parameter covariance.

# Function description
This routine is a visualization companion to [`Maranatha.FitConvergence.fit_convergence`](@ref). It produces a
publication-style convergence plot and saves it as a PNG file.

The ``x``-axis is ``h^2`` (with ``\\displaystyle{h = \\frac{b-a}{N}}`` supplied via `hs`), and the ``y``-axis is the
raw integral estimate ``I(h)`` with its pointwise error bar.

A convergence model is reconstructed from the fit parameters under the rule-dependent
leading power `p`:
```math
I(h) = I_0 + C_1 \\, h^p + C_2 \\, h^{p+2} + C_3 \\, h^{p+4} + \\ldots
```
where ``I_0`` is the extrapolated ``h \\to 0`` limit.

## Fit curve and uncertainty band
This function does **not** refit anything. It uses the stored fit output:

- `pvec = fit_result.params`
- `Cov  = fit_result.cov`

For each point on a dense grid in ``h``, it builds the basis vector
```math
\\varphi(h) =
\\begin{bmatrix}
1 & h^p & h^{p+2} & \\cdots
\\end{bmatrix}^{\\mathsf{T}}
```
and evaluates:
- fit curve: ``I_{\\text{fit}}(h) = \\bm{\\lambda} \\cdot \\bm{\\varphi}(h)``
- ``1 \\, \\sigma`` fit uncertainty (where ``V`` is covariance matrix):
```math
\\sigma_{\\text{fit}}(h)^2 = \\varphi(h)^{\\mathsf{T}} \\, V \\, \\varphi(h)
```

The plotted shaded band corresponds to ``I_{\\text{fit}}(h) \\pm \\sigma_{\\text{fit}}(h)``, and therefore includes
parameter correlations.

## Plot elements
The resulting figure contains:
- the fitted curve ``I_{\\text{fit}}(h)`` (line),
- the fit uncertainty band ``\\pm \\sigma`` (shaded region),
- the measured points with error bars,
- the extrapolated point at ``h^2 = 0`` with uncertainty `fit_result.estimate_error`.

The output file is saved as:

`convergence_<name>_<rule>.png`

# Arguments
- `name`: Label used in the output filename.
- `hs`: Step sizes `h` (typically ``\\displaystyle{h = \\frac{b-a}{N}}``).
- `estimates`: Quadrature estimates ``I(h)`` corresponding to `hs`.
- `errors`: Error estimates for ``I(h)`` (absolute values are used for plotting).
- `fit_result`: Fit object expected to provide:
  - `fit_result.params`
  - `fit_result.cov`
  - `fit_result.estimate`
  - `fit_result.estimate_error`

# Keyword arguments
- `rule`: Quadrature rule symbol used to determine the leading power `p` and
  to label the output filename.

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
        JobLoggerTools.error_benji("Unknown rule")

    # --- Input checks ---
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    # Raw x = h^2
    h2 = hs .^ 2
    errors_pos = abs.(errors)

    mask = (h2 .> 0) .& isfinite.(h2) .& isfinite.(estimates) .& isfinite.(errors_pos)

    h2p = h2[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    isempty(h2p) && JobLoggerTools.error_benji("No valid points to plot.")

    # --- New fit result structure ---
    pvec = fit_result.params
    I0      = fit_result.estimate
    I0_err  = fit_result.estimate_error

    # --- Build model automatically from params ---
    # Model: I(h) = I0 + C1*h^p + C2*h^(p+2) + ...
    Cov = fit_result.cov

    function basis_vec(h)
        v = Vector{Float64}(undef, length(pvec))
        v[1] = 1.0
        for i in 2:length(pvec)
            power = p + 2*(i-2)
            v[i] = h^power
        end
        return v
    end

    function model_and_err(h)
        φ = basis_vec(h)
        y = dot(pvec, φ)
        σ = sqrt(abs(dot(φ, Cov * φ)))
        return y, σ
    end

    # --- Smooth curve including extrapolated point at h^2 = 0 ---
    h2min = minimum(h2p)
    h2max = maximum(h2p)

    h2_range_log = 10 .^ range(log10(h2min), log10(h2max); length=200)

    # prepend zero explicitly
    h2_range = vcat(0.0, h2_range_log)

    # model needs h, not h^2
    h_range  = sqrt.(h2_range)
    y_fit = similar(h_range)
    y_err = similar(h_range)

    for i in eachindex(h_range)
        y_fit[i], y_err[i] = model_and_err(h_range[i])
    end


    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    # Fit curve
    ax.plot(h2_range, y_fit; color="black", linewidth=2.5)

    # --- Fit error band ---
    ax.fill_between(
        h2_range,
        y_fit .- y_err,
        y_fit .+ y_err;
        alpha=0.25,
        linewidth=0,
        color="black"
    )

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