# ============================================================================
# src/Documentation/Reporter/_fmt_avgerr_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_avgerr_tex(
        x::Real, 
        err::Real
    ) -> String

Format a central value and uncertainty for [``\\LaTeX``](https://www.latex-project.org/) output.

# Function description

This helper attempts to produce a compact parenthetical uncertainty string
using

    AvgErrFormatter.avgerr_e2d_from_float(...; latex_grouping=true)

Examples:

- `1.23(45)`
- `2.000\\,000\\,000(39)`

If the compact formatting fails for any reason, the function falls back to a
simple `x ± err` representation using [``\\LaTeX``](https://www.latex-project.org/)-compatible syntax:

    "%.7g \\pm %.2g"

# Arguments

- `x::Real`: Central value.
- `err::Real`: Uncertainty.

# Returns

- `String`: Formatted uncertainty string suitable for [``\\LaTeX``](https://www.latex-project.org/).

# Errors

- Formatting failures are caught internally; no exception is propagated.

# Notes

- Digit grouping may be applied to long mantissas.
- The fallback representation prioritizes robustness over compactness.
"""
function _fmt_avgerr_tex(
    x::Real, 
    err::Real
)
    return try
        AvgErrFormatter.avgerr_e2d_from_float(float(x), float(err); latex_grouping=true)
    catch
        @sprintf("%.7g \\pm %.2g", float(x), float(err))
    end
end