# ============================================================================
# src/Utils/MaranathaTOML/parse/_real_type_symbol_to_type.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _real_type_symbol_to_type(sym::Symbol) -> DataType

Map a supported `real_type` selector symbol to its concrete Julia scalar type.

# Function description

This helper resolves the symbolic `real_type` value used by the TOML
configuration layer into the concrete scalar type used for domain parsing.

It is currently used by [`parse_run_config_from_toml`](@ref) so that
string-valued domain literals can be parsed directly in the precision selected
by `[execution].real_type`.

# Arguments

- `sym::Symbol`: Scalar-type selector such as `:Float32`, `:Float64`,
  `:BigFloat`, or `:Double64`.

# Returns

- `DataType`: Concrete Julia scalar type corresponding to `sym`.

# Errors

- Throws if `sym` is not one of the supported `real_type` selectors.

# Notes

- `:Double64` is resolved to `DoubleFloats.Double64`.
- This helper performs selector resolution only; it does not parse values.
"""
@inline function _real_type_symbol_to_type(sym::Symbol)
    sym === :Float32  && return Float32
    sym === :Float64  && return Float64
    sym === :BigFloat && return BigFloat
    sym === :Double64 && return DoubleFloats.Double64
    error("Unsupported real_type=$(sym)")
end
