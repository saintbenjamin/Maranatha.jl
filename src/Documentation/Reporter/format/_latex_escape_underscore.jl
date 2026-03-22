# ============================================================================
# src/Documentation/Reporter/format/_latex_escape_underscore.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _latex_escape_underscore(
        s::AbstractString
    ) -> String

Escape underscores for safe use in [``\\LaTeX``](https://www.latex-project.org/) text mode.

# Function description

This helper replaces every underscore character `_` with the [``\\LaTeX``](https://www.latex-project.org/)-escaped
sequence `\\_`, preventing compilation errors in text contexts such as
captions, labels, or section titles.

# Arguments

- `s::AbstractString`: Input string.

# Returns

- `String`: String with underscores escaped for [``\\LaTeX``](https://www.latex-project.org/).

# Errors

- No explicit validation is performed.

# Notes

- Only underscores are handled; other special characters are not escaped.
- Intended for simple identifiers (file names, labels, rule names).
"""
function _latex_escape_underscore(
    s::AbstractString
)
    return replace(s, "_" => "\\_")
end