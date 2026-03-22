# ============================================================================
# src/Documentation/Reporter/format/_fmt_rule_boundary_cell_md.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_rule_boundary_cell_md(rule, boundary, d::Int, dim::Int) -> String

Format the Markdown rule/boundary cell for axis `d`.

# Function description
This helper resolves the axis-local rule and boundary entries and returns a
compact Markdown-friendly string of the form `"<rule> (<boundary>)"`.

# Arguments
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.
- `d::Int`: Axis index.
- `dim::Int`: Effective reporting dimension.

# Returns
- `String`: Markdown-ready rule/boundary cell text.

# Errors
- Propagates axis-resolution errors from [`_report_cfg_at`](@ref).
"""
@inline function _fmt_rule_boundary_cell_md(rule, boundary, d::Int, dim::Int)::String
    rd = _report_cfg_at(rule, d, dim)
    bd = _report_cfg_at(boundary, d, dim)
    return "$(string(rd)) ($(string(bd)))"
end