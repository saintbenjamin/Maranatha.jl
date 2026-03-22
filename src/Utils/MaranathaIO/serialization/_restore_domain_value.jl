# ============================================================================
# src/Utils/MaranathaIO/serialization/_restore_domain_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _restore_domain_value(x, T)

Reconstruct a domain value from a serialized representation.

# Function description

This routine reverses the transformation applied by
[`_storable_domain_value`](@ref), converting stored vector data back
into the domain representation expected by the computational code.

If the input is vector-like, the elements are converted to type `T`
and returned as a tuple. Scalar inputs are converted directly to `T`.

# Arguments

- `x`: Stored domain value.
- `T`: Target numeric type.

# Returns

- A domain value of type `T`:

  - `Tuple{T,...}` if `x` is vector-like,
  - scalar `T` otherwise.
"""
@inline function _restore_domain_value(x, T)
    if x isa AbstractVector
        return Tuple(convert.(T, x))
    else
        return convert(T, x)
    end
end
