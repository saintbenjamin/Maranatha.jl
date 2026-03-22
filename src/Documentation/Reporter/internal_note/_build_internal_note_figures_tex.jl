# ============================================================================
# src/Documentation/Reporter/internal_note/_build_internal_note_figures_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_internal_note_figures_tex(
        extrap_plot_name, 
        reldiff_plot_name; 
        title, 
        rule, 
        boundary
    ) -> String

Construct a [``\\LaTeX``](https://www.latex-project.org/) figure block for inclusion in the internal note.

# Function description

Generates a two-panel figure using subfigures:

- convergence fit and extrapolation,
- relative convergence error.

Captions are formatted using `\\texorpdfstring` to ensure compatibility
with PDF bookmarks while preserving [``\\LaTeX``](https://www.latex-project.org/) formatting in the document body.

# Arguments

- `extrap_plot_name`: Filename of the extrapolation plot.
- `reldiff_plot_name`: Filename of the relative-error plot.

# Keyword arguments

- `title`: Experiment identifier.
- `rule`: Quadrature rule.
- `boundary`: Boundary scheme.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/) code for a complete figure environment.

# Notes

- Underscores are escaped for [``\\LaTeX``](https://www.latex-project.org/) safety.
- Labels are generated using the raw (unescaped) title.
"""
function _build_internal_note_figures_tex(
    extrap_plot_name::AbstractString,
    reldiff_plot_name::AbstractString;
    title,
    rule,
    boundary,
)

    # --- LaTeX-safe (for document body) ---
    safe_title = _latex_escape_underscore(title)
    safe_rule = _latex_escape_underscore(string(rule))
    safe_boundary = boundary isa Symbol ?
        _latex_escape_underscore(String(boundary)) :
        "(" * join(_latex_escape_underscore.(String.(collect(boundary))), ",\\ ") * ")"

    boundary_str = if boundary isa Symbol
        String(boundary)
    elseif boundary isa Tuple || boundary isa AbstractVector
        join(String.(boundary), "_")
    else
        String(boundary)
    end

    # --- Plain text (for PDF bookmarks) ---
    plain_title = title
    plain_rule = string(rule)
    plain_boundary = boundary_str

    # --- Caption fragments with texorpdfstring ---
    cap_title = "\\texorpdfstring{\\texttt{$safe_title}}{$plain_title}"
    cap_rule  = "\\texorpdfstring{\\texttt{$safe_rule}}{$plain_rule}"
    cap_bound = "\\texorpdfstring{\\texttt{$safe_boundary}}{$plain_boundary}"

    io = IOBuffer()

    println(io, "% Auto-generated figure include file")
    println(io, "")
    println(io, "\\begin{figure}[htbp]")
    println(io, "  \\centering")
    println(io, "  \\subfigure[Convergence fit and extrapolated result.]{")
    println(io, "    \\includegraphics[width=0.48\\textwidth]{$(extrap_plot_name)}")
    println(io, "    \\label{fig:$(title)_extrap}")
    println(io, "  }")
    println(io, "  \\hfill")
    println(io, "  \\subfigure[Relative convergence error.]{")
    println(io, "    \\includegraphics[width=0.48\\textwidth]{$(reldiff_plot_name)}")
    println(io, "    \\label{fig:$(title)_reldiff}")
    println(io, "  }")
    println(io,
        "  \\caption{Summary plots for $(cap_title).}")
    println(io, "  \\label{fig:$(title)_summary}")
    println(io, "\\end{figure}")

    return String(take!(io))
end