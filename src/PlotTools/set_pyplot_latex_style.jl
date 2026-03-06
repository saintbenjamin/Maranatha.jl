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