# ============================================================================
# src/Utils/MaranathaTOML/validate/_validate_rule_spec_local.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _validate_rule_spec_local(rule, dim::Int) -> Nothing

Validate that a TOML-parsed rule specification is well formed for dimension
`dim`.

# Function description
This helper accepts either a scalar rule symbol shared across all axes or a
tuple/vector of per-axis rule symbols of length `dim`. Every resolved axis
entry is checked for supported family membership.

# Arguments
- `rule`: TOML-parsed rule specification.
- `dim::Int`: Expected problem dimension.

# Returns
- `nothing`

# Errors
- Throws if `dim < 1`, if an axis-wise rule specification has the wrong
  length, or if any axis-local rule is unsupported.
"""
@inline function _validate_rule_spec_local(rule, dim::Int)::Nothing
    dim >= 1 || error("Invalid dim: dim must be >= 1, but got dim=$(dim).")

    for d in 1:dim
        _rule_at_local(rule, d, dim)
    end

    return nothing
end
