# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_domain_scalar.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_domain_scalar(x, T)

Parse or convert one scalar domain endpoint into the target scalar type `T`.

# Function description

This helper is the scalar-level conversion routine used by
[`parse_run_config_from_toml`](@ref) for `[domain].a` and `[domain].b`.

Behavior depends on the input form:

- if `x isa AbstractString`, the string is stripped, parsed first as
  `BigFloat`, and then converted to `T`;
- otherwise, the value is converted directly via `T(x)`.

This allows string-valued TOML domain literals to preserve precision until the
selected `real_type` is known.

# Arguments

- `x`: Scalar domain value, typically either a numeric TOML value or a string
  literal containing a numeric value.
- `T`: Target scalar type.

# Returns

- Scalar endpoint represented in type `T`.

# Errors

- Throws if a string input cannot be parsed as a number.
- Throws if conversion to `T` fails.

# Notes

- This helper handles scalar values only. Collection-valued endpoints are
  handled by [`_parse_domain_endpoint`](@ref).
"""
@inline function _parse_domain_scalar(x, T)
    if x isa AbstractString
        return T(parse(BigFloat, strip(x)))
    else
        return T(x)
    end
end
