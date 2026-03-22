# ============================================================================
# src/Documentation/Reporter/_fmt_axis_interval_for_run_config.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _fmt_axis_interval_for_run_config(a, b, d::Int, dim::Int) -> String

Format the interval text for axis `d` in a run-configuration table.

# Function description
This helper resolves `a` and `b` on axis `d` and returns the plain interval
string `"(a_d, b_d)"`.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `d::Int`: Axis index.
- `dim::Int`: Effective reporting dimension.

# Returns
- `String`: Plain interval string for axis `d`.

# Errors
- Propagates axis-resolution errors from [`_report_cfg_at`](@ref).
"""
@inline function _fmt_axis_interval_for_run_config(a, b, d::Int, dim::Int)::String
    ad = _report_cfg_at(a, d, dim)
    bd = _report_cfg_at(b, d, dim)
    return "($(ad), $(bd))"
end