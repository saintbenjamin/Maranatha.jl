"""
    set_pyplot_latex_style(
        scale::Float64 = 0.5
    ) -> Nothing

Configure global [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) / [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) rendering parameters for publication-style figures.

This helper enables [``\\LaTeX``](https://www.latex-project.org/)-based text rendering and adjusts global plotting
settings such as font sizes, line widths, and marker sizes. It is typically used
internally by higher-level plotting routines in `Maranatha.PlotTools`.

# Arguments

`scale::Float64 = 0.5`
: Global scaling factor used when setting font sizes and related figure-style
  parameters.

# Returns

`nothing`.

# Errors

This function does not perform explicit argument validation, but downstream
[`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) / [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) configuration calls may fail if the plotting backend is not
available or if the local [``\\LaTeX``](https://www.latex-project.org/) environment is not configured properly.

# Notes

This function modifies global `matplotlib` state through `rcParams`, so its effect
persists for subsequently created figures in the current Julia session.
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