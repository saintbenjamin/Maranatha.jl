# Maranatha.Quadrature.QuadratureUtils

`Maranatha.Quadrature.QuadratureUtils` contains rule-agnostic helper utilities
shared across the quadrature subsystem of `Maranatha.jl`.

This module provides small building blocks used by multiple components,
especially for interpreting global configuration symbols (such as boundary
selectors) into forms suitable for local quadrature assembly. It does not
construct nodes, weights, or evaluate integrals directly.

---

## Responsibility in the quadrature layer

Within the quadrature architecture:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureUtils` | shared low-level helpers |
| `QuadratureNodes` | construct 1D nodes and weights |
| `QuadratureDispatch` | evaluate tensor-product integrals |

This module sits at the lowest level and provides functionality that is reused
by higher-level components.

---

## Overview

The utilities in this module are intentionally minimal and independent of
specific quadrature rules.

Currently, the module provides:

| Function | Responsibility |
|:--|:--|
| [`Maranatha.Quadrature.QuadratureUtils._decode_boundary`](@ref) | convert global boundary selectors into local endpoint types |
| [`Maranatha.Quadrature.QuadratureUtils._sanitize_nsamples_newton_cotes`](@ref) | adjust subdivision sequences to valid Newton-Cotes composite counts |

---

## Notes

- Functions in this module are lightweight and stateless.
- The leading underscore indicates internal helpers not intended for direct
  user-facing APIs.
- Additional shared utilities may be added here as the quadrature subsystem
  evolves.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureUtils,
]
Private = true
```