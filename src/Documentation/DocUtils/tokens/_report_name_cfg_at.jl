# ============================================================================
# src/Documentation/DocUtils/tokens/_report_name_cfg_at.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _report_name_cfg_at(x, d::Int, dim::Int)

Resolve the value of one report-metadata specification on axis `d`.

# Function description
Scalar inputs are treated as shared values and returned unchanged. Tuple/vector
inputs are validated against `dim` and indexed at axis `d`.

# Arguments
- `x`: Scalar or axis-wise report-metadata specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise inputs.

# Returns
- The scalar shared value or the axis-local entry `x[d]`.

# Errors
- Throws if an axis-wise specification has length different from `dim`.
"""
@inline function _report_name_cfg_at(x, d::Int, dim::Int)
    if _report_name_is_multi(x)
        length(x) == dim || error("Filename-spec mismatch: spec length must equal dim.")
        return x[d]
    end
    return x
end
