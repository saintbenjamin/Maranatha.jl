# ============================================================================
# src/Utils/MaranathaIO/summary/_toml_safe.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _toml_safe(x)

Convert an arbitrary object into a TOML-compatible representation.

# Function description

This recursive helper transforms complex Julia objects into values that
can be safely serialized in TOML format.

The conversion rules aim to preserve meaning while ensuring compatibility
with TOML's restricted data model.

Supported conversions include:

- Primitive TOML types (`Bool`, integers, strings) -> unchanged
- Floating-point values -> preserved if standard (`Float32`, `Float64`),
  otherwise converted to string
- Symbols -> converted to strings
- Tuples and vectors -> converted elementwise to arrays
- Dictionaries and named tuples -> converted to `Dict{String,Any}`
  with recursively processed values
- Other types -> converted to string representations

# Arguments

- `x`: Arbitrary Julia object.

# Returns

- A TOML-safe value composed only of supported scalar types,
  arrays, and string-keyed dictionaries.

# Notes

- This function prioritizes robustness over round-trip fidelity.
- Non-standard numeric types may be stringified to avoid loss of
  information during serialization.
"""
@inline function _toml_safe(x)
    if x isa Bool || x isa Integer || x isa AbstractString
        return x
    elseif x isa AbstractFloat
        return x isa Float32 || x isa Float64 ? x : string(x)
    elseif x isa Symbol
        return String(x)
    elseif x isa Tuple
        return [_toml_safe(v) for v in x]
    elseif x isa AbstractVector
        return [_toml_safe(v) for v in x]
    elseif x isa Dict
        out = Dict{String,Any}()
        for (k, v) in x
            out[string(k)] = _toml_safe(v)
        end
        return out
    elseif x isa NamedTuple
        out = Dict{String,Any}()
        for k in keys(x)
            out[string(k)] = _toml_safe(getproperty(x, k))
        end
        return out
    else
        return string(x)
    end
end
