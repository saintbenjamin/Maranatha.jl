# Maranatha.Quadrature.QuadratureNodes

`Maranatha.Quadrature.QuadratureNodes` provides the one-dimensional quadrature
node and weight generator used by all tensor-product integration backends in
`Maranatha.jl`.

This module constructs nodes and weights on an interval `[a, b]` according to a
symbolic rule specification. Higher-dimensional integrators reuse these 1D
components to build tensor-product quadrature schemes.

---

## Responsibility in the quadrature layer

Within the quadrature architecture:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureBoundarySpec` | boundary decoding and axis-wise boundary access |
| `QuadratureNodes` | construct 1D nodes and weights |
| `QuadratureDispatch` | perform multi-dimensional accumulation |

Thus, this module defines the geometric and weighting structure of a quadrature
rule, but does not execute the integration itself.

---

## Supported rule families

The generator currently supports multiple rule families:

- Composite Newton–Cotes rules (`:newton_p*`)
- Composite Gauss-family rules (`:gauss_p*`)
- B-spline-based quadrature rules (`:bspline_*`)

Each family may impose its own constraints on parameters such as boundary
conditions or smoothing options.

---

## Overview

The module exposes a single primary entry point:

| Function | Responsibility |
|:--|:--|
| [`Maranatha.Quadrature.QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref) | construct nodes and weights on `[a,b]` |

The returned nodes and weights are typically reused across dimensions by
tensor-product quadrature drivers.

---

## Notes

- Boundary-condition semantics are interpreted via
  [`Maranatha.Utils.QuadratureBoundarySpec._decode_boundary`](@ref).
- The module returns floating-point nodes and weights suitable for numerical
  integration.
- Rule-specific logic is delegated to the corresponding backend modules
  (Newton–Cotes, Gauss, B-spline).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureNodes,
]
Private = true
```
