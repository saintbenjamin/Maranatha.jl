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
| [`_sanitize_nsamples_newton_cotes`](@ref) | adjust subdivision sequences to satisfy Newton-Cotes tiling constraints |

# Notes

- Functions in this module are intended to be lightweight and reusable.
- The leading underscore indicates internal helpers not meant for direct user
  calls, although they may be referenced by multiple internal modules.
- Additional shared utilities may be added here as the quadrature system grows.
"""
module QuadratureUtils

import ..JobLoggerTools
import ..Quadrature

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

"""
    _sanitize_nsamples_newton_cotes(
        nsamples::Vector{Int},
        rule::Symbol,
        boundary::Symbol
    ) -> Vector{Int}

Sanitize a candidate subdivision sequence for Newton-Cotes composite rules.

# Function description
Newton-Cotes composite formulas do not accept arbitrary subdivision counts.
For a given local node count `p` and boundary pattern `boundary`, valid values
must satisfy the tiling constraint

```math
N_{\\mathrm{sub}} = w_L + m (p-1) + w_R,
```

where `w_L` and `w_R` are the boundary block widths and `m ≥ 0` is an integer.

This helper transforms an arbitrary input sequence into a valid sequence that:

* preserves the original length,
* forms a valid arithmetic progression with step `(p - 1)`,
* starts from the nearest admissible value not exceeding the first input
  element, or from the smallest admissible value if none is smaller.

If `rule` is not a Newton-Cotes rule, the input is returned unchanged.

# Arguments

* `nsamples::Vector{Int}`
  Candidate subdivision counts supplied by the caller.

* `rule::Symbol`
  Quadrature rule selector. Must be of the form `:newton_pK` to activate
  Newton-Cotes sanitization.

* `boundary::Symbol`
  Boundary pattern symbol determining the left and right endpoint types.

# Returns

* `Vector{Int}`
  A corrected subdivision sequence compatible with the Newton-Cotes composite
  tiling constraint. The returned vector has the same length as `nsamples`.

# Errors

* Propagates validation errors from
  `NewtonCotes._parse_newton_p`,
  [`_decode_boundary`](@ref), and
  `NewtonCotes._local_width`.
* Does not throw for invalid subdivision counts; instead, it adjusts them.

# Notes

* This helper is intended for internal use by runner-level components that
  accept user-supplied subdivision arrays.
* A warning is emitted if the sequence is modified.
* The resulting sequence always represents a monotone refinement ladder with
  constant step `(p - 1)`.
"""
function _sanitize_nsamples_newton_cotes(
    nsamples::Vector{Int},
    rule::Symbol,
    boundary::Symbol
)::Vector{Int}

    # Not Newton-Cotes → no change
    if !Quadrature.NewtonCotes._is_newton_cotes_rule(rule)
        return nsamples
    end

    isempty(nsamples) && return nsamples

    p = Quadrature.NewtonCotes._parse_newton_p(rule)

    Ltype, Rtype = _decode_boundary(boundary)
    wL = Quadrature.NewtonCotes._local_width(p, Ltype)
    wR = Quadrature.NewtonCotes._local_width(p, Rtype)

    step = p - 1
    base = wL + wR   # smallest valid N

    N0 = first(nsamples)

    # ---- nearest valid ≤ N0 ----
    if N0 >= base
        m = (N0 - base) ÷ step
        start = base + m * step
    else
        start = base
    end

    # if start > N0 (only possible when N0 < base)
    if start < base
        start = base
    end

    # ---- build arithmetic progression ----
    L = length(nsamples)
    newN = [start + (i-1)*step for i in 1:L]

    if newN != nsamples
        JobLoggerTools.warn_benji(
            "nsamples corrected for $rule, $boundary\n" *
            "input = $(nsamples)\n" *
            "using = $(newN)"
        )
    end

    return newN
end

end  # module QuadratureUtils