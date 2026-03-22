# ============================================================================
# src/Utils/MaranathaIO/serialization/_restore_rule_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _restore_rule_value(x)

Reconstruct a quadrature-rule specification from serialized storage form.

# Function description
String input is converted back to a scalar `Symbol`. Vector input is converted
to a tuple of `Symbol` values, restoring the axis-wise rule form used by the
runtime pipeline.

# Arguments
- `x`: Serialized rule representation.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Conversion errors propagate if the stored representation is malformed.
"""
@inline function _restore_rule_value(x)
    if x isa AbstractString
        return Symbol(x)
    elseif x isa AbstractVector
        return Tuple(Symbol.(x))
    else
        return Symbol(x)
    end
end
