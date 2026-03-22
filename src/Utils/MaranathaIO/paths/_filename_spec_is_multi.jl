# ============================================================================
# src/Utils/MaranathaIO/paths/_filename_spec_is_multi.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _filename_spec_is_multi(x) -> Bool

Return `true` if `x` participates as an axis-wise filename specification.

# Function description
Tuple and vector values are classified as axis-wise metadata, while all other
values are treated as shared scalar metadata for filename construction.

# Arguments
- `x`: Candidate filename-metadata value.

# Returns
- `Bool`: `true` for tuple/vector input, `false` otherwise.

# Errors
- No explicit validation is performed.
"""
@inline _filename_spec_is_multi(x) = x isa Tuple || x isa AbstractVector
