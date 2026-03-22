# ============================================================================
# src/Documentation/Reporter/_fmt_tex_texttt_sci.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_tex_texttt_sci(
        x::Real
    ) -> String

Format a number in [``\\LaTeX``](https://www.latex-project.org/) monospace scientific notation.

# Function description

This helper wraps the scientific-notation output of
[`_fmt_sci_texttt`](@ref) inside a [``\\LaTeX``](https://www.latex-project.org/) `\\texttt{}` command:

    \\texttt{1.234567e-03}

It is intended for inclusion in [``\\LaTeX``](https://www.latex-project.org/) tables or inline text where a
monospaced numeric appearance improves readability.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/)-formatted scientific-notation string.

# Errors

- No explicit validation is performed.

# Notes

- The returned string is safe for use in [``\\LaTeX``](https://www.latex-project.org/) document bodies.
- No math-mode delimiters (`\$`) are added automatically.
"""
function _fmt_tex_texttt_sci(
    x::Real
)
    return "\\texttt{$(_fmt_sci_texttt(x))}"
end