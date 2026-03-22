# ============================================================================
# src/Documentation/Reporter/format/_fmt_axis_cell_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_axis_cell_tex(a, b, d::Int, dim::Int) -> String

Format the LaTeX axis-cell label for axis `d`.

# Function description
This helper combines the axis name `x_d` with a LaTeX-escaped interval string
used in run-configuration tables.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `d::Int`: Axis index.
- `dim::Int`: Effective reporting dimension.

# Returns
- `String`: LaTeX-ready axis label.

# Errors
- Propagates interval-formatting errors from
  [`_fmt_axis_interval_for_run_config`](@ref).
"""
@inline function _fmt_axis_cell_tex(a, b, d::Int, dim::Int)::String
    interval_txt = _latex_escape_underscore(
        _fmt_axis_interval_for_run_config(a, b, d, dim)
    )
    return "\$x_{$d}\$: \\texttt{$interval_txt}"
end