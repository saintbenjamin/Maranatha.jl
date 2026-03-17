# ============================================================================
# src/Documentation/Reporter/_fmt_md_code_sci.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_md_code_sci(x::Real) -> String

Format a number as Markdown inline code in scientific notation.

# Function description

This helper wraps the scientific-notation output of
[`_fmt_sci_texttt`](@ref) inside Markdown backticks:

    `1.234567e-03`

It is suitable for README files, reports, or logs rendered using
Markdown engines.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: Markdown inline-code representation.

# Errors

- No explicit validation is performed.

# Notes

- Intended for human-readable documentation rather than machine parsing.
"""
function _fmt_md_code_sci(x::Real)
    return "`$(_fmt_sci_texttt(x))`"
end