# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_rule_spec.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_rule_spec(x)

Parse a scalar or axis-wise quadrature-rule specification from TOML input.

# Function description
Scalar input is parsed as one rule symbol. Tuple/vector input is parsed
entrywise and returned as a tuple of rule symbols.

# Arguments
- `x`: TOML rule value, scalar or array-like.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Propagates entry-level parsing errors from [`_parse_rule_entry`](@ref).
"""
@inline function _parse_rule_spec(x)
    if x isa Tuple
        return Tuple(_parse_rule_entry(v) for v in x)
    elseif x isa AbstractVector
        return Tuple(_parse_rule_entry(v) for v in x)
    else
        return _parse_rule_entry(x)
    end
end
