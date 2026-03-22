# ============================================================================
# src/Documentation/Reporter/run_config/_report_cfg_is_multi.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _report_cfg_is_multi(x) -> Bool

Return `true` if `x` is treated as an axis-wise reporting specification.

# Function description
Tuple and vector values are classified as per-axis metadata, while all other
inputs are treated as shared scalar values.

# Arguments
- `x`: Candidate reporting-metadata value.

# Returns
- `Bool`: `true` for tuple/vector input, `false` otherwise.

# Errors
- No explicit validation is performed.
"""
@inline _report_cfg_is_multi(x) = x isa Tuple || x isa AbstractVector