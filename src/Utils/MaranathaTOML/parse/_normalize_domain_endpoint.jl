# ============================================================================
# src/Utils/MaranathaTOML/parse/_normalize_domain_endpoint.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _normalize_domain_endpoint(x)

Normalize a domain endpoint into the internal storage form used by this module.

# Function description

This helper converts tuple- and vector-like endpoints into a freshly
allocated `Vector` via `collect`. Scalar values are returned unchanged.

The result is used by [`parse_run_config_from_toml`](@ref) so that
collection-valued endpoints are stored uniformly before later validation.
# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).

# Returns

- `collect(x)` if `x isa Tuple`
- `collect(x)` if `x isa AbstractVector`
- `x` otherwise
"""
@inline function _normalize_domain_endpoint(x)
    if x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return x
    end
end
