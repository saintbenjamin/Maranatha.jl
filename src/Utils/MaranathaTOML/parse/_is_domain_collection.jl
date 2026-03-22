# ============================================================================
# src/Utils/MaranathaTOML/parse/_is_domain_collection.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _is_domain_collection(x) -> Bool

Return `true` if `x` is treated as a collection-valued domain endpoint.

# Function description

This predicate is the internal domain-style classifier used by the TOML
parsing and validation pipeline.

The current implementation returns `true` only for:

- `Tuple`
- `AbstractVector`

All other values are treated as scalar endpoints and return `false`.

# Arguments

- `x`: Domain endpoint candidate.

# Returns

- `Bool`: `true` if `x` is a tuple or vector, `false` otherwise.
"""
@inline function _is_domain_collection(x)
    return x isa Tuple || x isa AbstractVector
end
