# ============================================================================
# src/Quadrature/QuadratureUtils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureUtils

Common utility helpers shared across quadrature backends.

# Module description
`QuadratureUtils` provides small, rule-independent helper functions used by
multiple components of the quadrature layer. These utilities do not construct
nodes, weights, or perform integration themselves; instead, they implement
shared logic that would otherwise be duplicated across backends.

Currently, the module focuses on interpreting boundary-condition selectors and
mapping them to local endpoint semantics used by composite quadrature rules.

# Responsibility in the quadrature layer

Within the overall architecture:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureUtils` | shared rule-agnostic helpers |
| `QuadratureNodes` | construct 1D nodes and weights |
| Quadrature backends | evaluate tensor-product sums |

This module sits at the lowest level of the quadrature stack and has no
dependencies on specific rule families.

# Overview

At present, the module provides:

| Function | Responsibility |
|:--|:--|
| [`_decode_boundary`](@ref) | map global boundary selectors to local endpoint types |

# Notes

- Functions in this module are intended to be lightweight and reusable.
- The leading underscore indicates internal helpers not meant for direct user
  calls, although they may be referenced by multiple internal modules.
- Additional shared utilities may be added here as the quadrature system grows.
"""
module QuadratureUtils

import ..JobLoggerTools

"""
    _decode_boundary(
        boundary::Symbol
    ) -> Tuple{Symbol,Symbol}

Decode a composite boundary selector into left/right local endpoint kinds.

# Function description
This helper maps the global boundary pattern into a pair of local endpoint tags
used by the Newton-Cotes composite assembly:

- `:closed` means the local block includes the endpoint node.
- `:opened` means the local block uses the shifted open-type construction.

Supported patterns are:

- `:LU_ININ` -> `(:closed, :closed)`
- `:LU_EXIN` -> `(:opened, :closed)`
- `:LU_INEX` -> `(:closed, :opened)`
- `:LU_EXEX` -> `(:opened, :opened)`

# Arguments
- `boundary`: Boundary pattern symbol.

# Returns
- `Tuple{Symbol,Symbol}`: `(Ltype, Rtype)`, each equal to `:closed` or `:opened`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is not one of
  `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, or `:LU_EXEX`.
"""
@inline function _decode_boundary(
    boundary::Symbol
)
    if boundary === :LU_ININ
        return (:closed, :closed)
    elseif boundary === :LU_EXIN
        return (:opened, :closed)
    elseif boundary === :LU_INEX
        return (:closed, :opened)
    elseif boundary === :LU_EXEX
        return (:opened, :opened)
    else
        JobLoggerTools.error_benji("boundary must be one of: :LU_ININ | :LU_EXIN | :LU_INEX | :LU_EXEX (got $boundary)")
    end
end


end  # module QuadratureUtils