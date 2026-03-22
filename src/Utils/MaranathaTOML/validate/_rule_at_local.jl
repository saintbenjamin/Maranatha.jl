# ============================================================================
# src/Utils/MaranathaTOML/validate/_rule_at_local.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _rule_at_local(rule, d::Int, dim::Int) -> Symbol

Resolve the scalar TOML-parsed rule symbol used on axis `d`.

# Function description
Scalar rules are returned unchanged after validation. Tuple/vector rules are
validated against `dim`, indexed at `d`, and checked for supported family
membership.

# Arguments
- `rule`: Scalar or axis-wise TOML-parsed rule specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise input.

# Returns
- `Symbol`: Scalar rule symbol used on axis `d`.

# Errors
- Throws if the rule specification is malformed or unsupported.
"""
@inline function _rule_at_local(rule, d::Int, dim::Int)::Symbol
    if rule isa Symbol
        _rule_family_local(rule)
        return rule
    elseif rule isa Tuple || rule isa AbstractVector
        length(rule) == dim || error(
            "Invalid rule specification: length(rule) must equal dim=$(dim)."
        )
        rd = rule[d]
        rd isa Symbol || error(
            "Invalid rule specification: rule[$d] must be a Symbol."
        )
        _rule_family_local(rd)
        return rd
    else
        error(
            "Invalid rule specification: expected Symbol or tuple/vector of Symbols, got $(typeof(rule))."
        )
    end
end
