# ============================================================================
# src/Utils/MaranathaTOML/parse/_parse_boundary_entry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _parse_boundary_entry(x) -> Symbol

Parse one scalar boundary entry from TOML input.

# Function description
This helper accepts either a `Symbol` or a string-like TOML value, normalizes
it to `Symbol`, and validates it against [`QuadratureBoundarySpec._decode_boundary`](@ref).

# Arguments
- `x`: Scalar boundary entry from TOML input.

# Returns
- `Symbol`: Validated boundary symbol.

# Errors
- Throws if `x` is neither a symbol nor a string.
- Throws if the decoded boundary symbol is unsupported.
"""
@inline function _parse_boundary_entry(x)::Symbol
    if x isa Symbol
        QuadratureBoundarySpec._decode_boundary(x)
        return x
    elseif x isa AbstractString
        s = Symbol(strip(x))
        QuadratureBoundarySpec._decode_boundary(s)
        return s
    else
        error("Invalid boundary entry: expected Symbol or String, got $(typeof(x)).")
    end
end
