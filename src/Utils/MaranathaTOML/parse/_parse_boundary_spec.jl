# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_boundary_spec.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_boundary_spec(x)

Parse a scalar or axis-wise boundary specification from TOML input.

# Function description
Scalar input is parsed as one boundary entry. Tuple/vector input is parsed
entrywise and returned as a tuple of boundary symbols.

# Arguments
- `x`: TOML boundary value, scalar or array-like.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Propagates entry-level parsing errors from [`_parse_boundary_entry`](@ref).
"""
@inline function _parse_boundary_spec(x)
    if x isa Tuple
        vals = Tuple(_parse_boundary_entry(v) for v in x)
        return vals
    elseif x isa AbstractVector
        vals = Tuple(_parse_boundary_entry(v) for v in x)
        return vals
    else
        return _parse_boundary_entry(x)
    end
end
