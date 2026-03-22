# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_rule_entry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_rule_entry(x) -> Symbol

Parse one scalar quadrature-rule entry from TOML input.

# Function description
This helper accepts either a `Symbol` or a string-like TOML value and
normalizes it to `Symbol`.

# Arguments
- `x`: Scalar rule entry from TOML input.

# Returns
- `Symbol`: Parsed rule symbol.

# Errors
- Throws if `x` is neither a symbol nor a string.
"""
@inline function _parse_rule_entry(x)::Symbol
    if x isa Symbol
        return x
    elseif x isa AbstractString
        return Symbol(strip(x))
    else
        error("Invalid rule entry: expected Symbol or String, got $(typeof(x)).")
    end
end
