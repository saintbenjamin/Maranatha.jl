# ============================================================================
# src/Documentation/Reporter/internal_note/_build_internal_note_master_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_internal_note_master_tex(
        summary_tex_name, 
        fig_tex_name; 
        title, 
        rule, 
        boundary, 
        author, 
        affiliation, 
        abstract_text
    ) -> String

Construct the master REVTeX document for the internal note.

# Function description

Creates a complete [``\\LaTeX``](https://www.latex-project.org/) document including:

- REVTeX class configuration,
- required packages,
- a title block derived from `rule` and `boundary`,
- optional author and affiliation blocks,
- an optional abstract,
- figure and summary inclusions.

The resulting document compiles independently.

# Arguments

- `summary_tex_name`: Filename of the summary table file.
- `fig_tex_name`: Filename of the figure include file.

# Keyword arguments

- `title`: Experiment identifier accepted by the helper interface. The current
  implementation does not embed it in the generated `\\title{...}` string.
- `rule`: Quadrature rule.
- `boundary`: Boundary scheme.
- `author`: Author name (optional).
- `affiliation`: Author affiliation (optional).
- `abstract_text`: Abstract text (optional).

# Returns

- `String`: Complete [``\\LaTeX``](https://www.latex-project.org/) document source.

# Notes

- Uses `revtex4-2` with preprint-style options.
- Figure directory is assumed to be `./figs/`.
- Rule and boundary strings are LaTeX-escaped before being placed in the title.
"""
function _build_internal_note_master_tex(
    summary_tex_name::AbstractString,
    fig_tex_name::AbstractString;
    title,
    rule,
    boundary,
    author::Union{Nothing,AbstractString} = nothing,
    affiliation::Union{Nothing,AbstractString} = nothing,
    abstract_text::Union{Nothing,AbstractString} = nothing,
)
    # safe_title = _latex_escape_underscore(title)
    safe_rule = _latex_escape_underscore(string(rule))
    safe_boundary = boundary isa Symbol ?
        _latex_escape_underscore(String(boundary)) :
        "(" * join(_latex_escape_underscore.(String.(collect(boundary))), ",\\ ") * ")"

    # note_title = "Maranatha.jl: \\texttt{$safe_title}, " *
    #              "\\texttt{$safe_rule}, " *
    #             "\\texttt{$safe_boundary}"
    note_title = "Maranatha.jl: \\texttt{$safe_rule}, " *
                 "\\texttt{$safe_boundary}"


    author_block = isnothing(author) ? "" : "\\author{$author}\n"
    affiliation_block = isnothing(affiliation) ? "" : "\\affiliation{$affiliation}\n"

    abstract_block = isnothing(abstract_text) ? "" : """
\\begin{abstract}
$abstract_text
\\end{abstract}
"""

    return """
%
%\\documentclass[12pt]{article}
\\documentclass[prd,onecolumn,12pt,showpacs,showkeys,preprintnumbers,floatfix,nofootinbib,superscriptaddress]{revtex4-2}

%------------------
% used packages
%------------------
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{subfigure}
\\usepackage{array}
\\usepackage{dcolumn}
\\usepackage{bm}
\\usepackage{bbold}
\\usepackage{latexsym}
\\usepackage{longtable}
\\usepackage{hyperref}
\\usepackage{mathrsfs}
\\usepackage{color}
\\usepackage{resizegather}
\\usepackage{cancel}
\\usepackage{mathtools}
\\usepackage{comment}
\\usepackage{csquotes}

\\graphicspath{{./figs/}}

%--------------------------------------------
% allow page break in the middle of equations
%--------------------------------------------
\\allowdisplaybreaks

\\begin{document}

\\title{$note_title}
$author_block$affiliation_block\\date{\\today}
$abstract_block\\maketitle

\\input{$fig_tex_name}
\\input{$summary_tex_name}

\\end{document}
"""
end