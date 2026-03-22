# ============================================================================
# src/Documentation/DocUtils/tokens/_report_name_is_multi.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _report_name_is_multi(x) -> Bool

Return `true` if `x` is treated as an axis-wise specification for report-name
construction.

# Function description
This helper classifies tuple and vector values as multi-axis specifications and
all other values as shared scalar specifications.

# Arguments
- `x`: Candidate report-metadata value.

# Returns
- `Bool`: `true` for tuple/vector input, `false` otherwise.

# Errors
- No explicit validation is performed.
"""
@inline _report_name_is_multi(x) = x isa Tuple || x isa AbstractVector
