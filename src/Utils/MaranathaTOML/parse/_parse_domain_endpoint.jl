# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_domain_endpoint.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_domain_endpoint(x, T)

Parse a scalar or collection-valued domain endpoint into the target scalar type `T`.

# Function description

This helper lifts [`_parse_domain_scalar`](@ref) to the endpoint forms accepted
by the TOML configuration layer.

Behavior depends on `x`:

- if `x isa Tuple`, each element is parsed / converted and the result is
  returned as a new vector;
- if `x isa AbstractVector`, each element is parsed / converted and the result
  is returned as a new vector;
- otherwise, `x` is treated as a scalar endpoint and converted directly.

# Arguments

- `x`: Domain endpoint given either as a scalar value or an array-like
  collection of per-axis bounds.
- `T`: Target scalar type.

# Returns

- Parsed scalar `T` for scalar input.
- `Vector{T}` for tuple- or vector-like input.

# Errors

- Propagates parsing or conversion errors from [`_parse_domain_scalar`](@ref).

# Notes

- This helper accepts both numeric TOML values and string-valued numeric
  literals.
- Tuple input is normalized to a vector so downstream endpoint handling can use
  one collection representation.
"""
@inline function _parse_domain_endpoint(x, T)
    if x isa Tuple
        return [_parse_domain_scalar(v, T) for v in x]
    elseif x isa AbstractVector
        return [_parse_domain_scalar(v, T) for v in x]
    else
        return _parse_domain_scalar(x, T)
    end
end
