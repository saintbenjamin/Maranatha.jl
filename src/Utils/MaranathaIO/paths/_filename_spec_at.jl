# ============================================================================
# src/Utils/MaranathaIO/paths/_filename_spec_at.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _filename_spec_at(x, d::Int, dim::Int)

Resolve one filename-metadata value on axis `d`.

# Function description
Scalar inputs are treated as shared values and returned unchanged. Tuple/vector
inputs are validated against `dim` and indexed at axis `d`.

# Arguments
- `x`: Scalar or axis-wise filename-metadata specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise inputs.

# Returns
- The scalar shared value or the axis-local entry `x[d]`.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if an axis-wise input has
  length different from `dim`.
"""
@inline function _filename_spec_at(x, d::Int, dim::Int)
    if _filename_spec_is_multi(x)
        length(x) == dim || JobLoggerTools.error_benji(
            "Filename-spec mismatch: spec length must equal dim."
        )
        return x[d]
    end
    return x
end
