# ============================================================================
# src/Documentation/Reporter/format/_fmt_rule_boundary_cell_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_rule_boundary_cell_tex(rule, boundary, d::Int, dim::Int) -> String

Format the LaTeX rule/boundary cell for axis `d`.

# Function description
This helper resolves the axis-local rule and boundary entries, escapes them for
LaTeX output, and returns a compact string suitable for run-configuration
tables.

# Arguments
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.
- `d::Int`: Axis index.
- `dim::Int`: Effective reporting dimension.

# Returns
- `String`: LaTeX-ready rule/boundary cell text.

# Errors
- Propagates axis-resolution errors from [`_report_cfg_at`](@ref).
"""
@inline function _fmt_rule_boundary_cell_tex(rule, boundary, d::Int, dim::Int)::String
    safe_rule = _latex_escape_underscore(string(_report_cfg_at(rule, d, dim)))
    safe_boundary = _latex_escape_underscore(string(_report_cfg_at(boundary, d, dim)))
    return "\\texttt{$safe_rule} (\\texttt{$safe_boundary})"
end