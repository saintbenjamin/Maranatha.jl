# ============================================================================
# src/Documentation/Reporter/_fmt_avgerr_md.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_avgerr_md(
        x::Real, 
        err::Real
    ) -> String

Format a central value and uncertainty for Markdown output.

# Function description

This helper is analogous to [`_fmt_avgerr_tex`](@ref) but produces
Markdown-friendly output.

It first attempts compact parenthetical formatting via

    AvgErrFormatter.avgerr_e2d_from_float(...; latex_grouping=false)

If that fails, it falls back to a Unicode plus–minus representation:

    "%.7g ± %.2g"

# Arguments

- `x::Real`: Central value.
- `err::Real`: Uncertainty.

# Returns

- `String`: Formatted uncertainty string suitable for Markdown text.

# Errors

- No exception is propagated from formatting failures.

# Notes

- Digit grouping is disabled to avoid introducing [``\\LaTeX``](https://www.latex-project.org/)-specific markup.
- Intended for reports, README files, or console-rendered Markdown.
"""
function _fmt_avgerr_md(
    x::Real, 
    err::Real
)
    return try
        AvgErrFormatter.avgerr_e2d_from_float(float(x), float(err); latex_grouping=false)
    catch
        @sprintf("%.7g ± %.2g", float(x), float(err))
    end
end