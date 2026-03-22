# ============================================================================
# src/Utils/MaranathaTOML/validate/_rule_family_local.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _rule_family_local(rule::Symbol) -> Symbol

Classify a TOML-parsed scalar rule symbol by family.

# Function description
This helper performs a lightweight local family classification used during TOML
validation, without depending on the quadrature module's internal helpers.

# Arguments
- `rule::Symbol`: Scalar rule symbol to classify.

# Returns
- `Symbol`: One of `:newton_cotes`, `:gauss`, or `:bspline`.

# Errors
- Throws if `rule` is not a supported rule symbol.
"""
@inline function _rule_family_local(rule::Symbol)::Symbol
    s = String(rule)

    startswith(s, "newton_p") && return :newton_cotes
    startswith(s, "gauss_p") && return :gauss

    if startswith(s, "bspline_interp_p") || startswith(s, "bspline_smooth_p")
        return :bspline
    end

    error("Invalid rule specification: unsupported rule symbol $(rule).")
end
