# ============================================================================
# src/Documentation/Reporter/_fmt_sci_texttt.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_sci_texttt(x::Real) -> String

Format a real number in fixed scientific notation.

# Function description

This helper converts `x` to a floating-point value and formats it using
six-digit scientific notation:

    "%.6e"

It is primarily used as a low-level building block for text-based reporting,
where a consistent machine-readable numeric format is required.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: Scientific-notation representation of `x`.

# Errors

- No explicit validation is performed.
- Non-finite values (`NaN`, `Inf`) are passed through to `Printf`.

# Notes

- The output does **not** include any [``\\LaTeX``](https://www.latex-project.org/) or Markdown markup.
- Intended for internal use by higher-level formatting helpers.
"""
function _fmt_sci_texttt(x::Real)
    return @sprintf("%.6e", float(x))
end