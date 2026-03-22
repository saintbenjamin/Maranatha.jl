# ============================================================================
# src/Utils/MaranathaIO/serialization/_storable_boundary_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _storable_boundary_value(x)

Convert a boundary specification into a serialization-friendly representation.

# Function description
Scalar boundary symbols are converted to strings. Tuple/vector axis-wise
boundary specifications are converted to vectors of strings. Other inputs are
returned unchanged.

# Arguments
- `x`: Boundary specification to normalize for storage.

# Returns
- A storage-friendly boundary representation composed of strings and standard
  containers.

# Errors
- No explicit validation is performed.
"""
@inline function _storable_boundary_value(x)
    if x isa Symbol
        return String(x)
    elseif x isa Tuple
        return String.(collect(x))
    elseif x isa AbstractVector
        return String.(collect(x))
    else
        return x
    end
end
