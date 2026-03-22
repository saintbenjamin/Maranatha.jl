# ============================================================================
# src/Utils/MaranathaIO/serialization/_is_scalar_domain_value.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _is_scalar_domain_value(x) -> Bool

Return `true` if `x` represents a scalar domain value.

# Function description

This predicate distinguishes scalar endpoints from multi-axis domain
representations. A value is considered scalar if it is neither a `Tuple`
nor an `AbstractVector`.

It is typically used when determining whether a problem domain is
one-dimensional (scalar) or multi-dimensional (tuple/vector-based).

# Arguments

- `x`: Domain endpoint candidate.

# Returns

- `Bool`: `true` if `x` is scalar-like, `false` if it represents a
  multi-component domain.
"""
@inline _is_scalar_domain_value(x) = !(x isa Tuple) && !(x isa AbstractVector)
