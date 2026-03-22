# ============================================================================
# src/Utils/MaranathaTOML/validate/_validate_refinement_rule_family_local.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _validate_refinement_rule_family_local(
        rule,
        dim::Int,
        err_method::Symbol,
    ) -> Nothing

Validate the current refinement restriction on axis-wise TOML rule specs.

# Function description
When `err_method != :refinement`, this helper returns immediately. For
refinement runs, it verifies that all axis-local rule entries belong to one
common quadrature family.

# Arguments
- `rule`: TOML-parsed rule specification.
- `dim::Int`: Expected problem dimension.
- `err_method::Symbol`: Parsed error-method selector.

# Returns
- `nothing`

# Errors
- Throws if `err_method == :refinement` and the resolved per-axis rules do not
  all belong to one family.
- Propagates rule-validation errors from [`_rule_at_local`](@ref) and
  [`_rule_family_local`](@ref).
"""
@inline function _validate_refinement_rule_family_local(
    rule,
    dim::Int,
    err_method::Symbol,
)::Nothing
    err_method === :refinement || return nothing

    fam = _rule_family_local(_rule_at_local(rule, 1, dim))
    for d in 2:dim
        fam_d = _rule_family_local(_rule_at_local(rule, d, dim))
        fam_d == fam || error(
            "Invalid rule specification: err_method=:refinement requires all axis-wise rules to belong to one family, but got rule=$(rule)."
        )
    end

    return nothing
end
