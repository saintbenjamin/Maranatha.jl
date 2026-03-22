# ============================================================================
# src/Utils/MaranathaIO/serialization/_storable_rule_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _storable_rule_value(x)

Convert a quadrature-rule specification into a serialization-friendly
representation.

# Function description
Scalar rule symbols are converted to strings. Tuple/vector axis-wise rule
specifications are converted to vectors of strings. Other inputs are returned
unchanged.

# Arguments
- `x`: Rule specification to normalize for storage.

# Returns
- A storage-friendly rule representation composed of strings and standard
  containers.

# Errors
- No explicit validation is performed.
"""
@inline function _storable_rule_value(x)
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
