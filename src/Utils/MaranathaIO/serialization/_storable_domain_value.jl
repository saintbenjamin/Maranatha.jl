# ============================================================================
# src/Utils/MaranathaIO/serialization/_storable_domain_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _storable_domain_value(x)

Convert a domain value into a container suitable for serialization.

# Function description

This helper normalizes domain endpoints for storage (e.g., in TOML or JSON)
by converting tuple- or vector-like objects into concrete vectors while
leaving scalar values unchanged.

The goal is to ensure that serialized representations use standard,
portable container types.

# Arguments

- `x`: Domain value (scalar, tuple, or vector-like).

# Returns

- A storable representation:

  - `Vector` if `x` is a tuple or vector-like object,
  - the original value if `x` is scalar.
"""
@inline function _storable_domain_value(x)
    if x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return x
    end
end
