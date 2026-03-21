# Maranatha.Utils.QuadratureBoundarySpec

`Maranatha.Utils.QuadratureBoundarySpec` centralizes parsing, validation, axis-wise
selection, and local endpoint decoding for the boundary specification used
throughout the quadrature and error-estimation layers.

---

## Overview

Boundary specifications in `Maranatha.jl` may be supplied either as:

- a single scalar symbol shared by all axes, or
- a tuple / vector of per-axis boundary symbols.

`QuadratureBoundarySpec` provides the shared helper layer that makes those forms usable
throughout the package.

The currently supported scalar boundary symbols are:

- `:LU_ININ`
- `:LU_EXIN`
- `:LU_INEX`
- `:LU_EXEX`

---

## Main responsibilities

| Helper | Responsibility |
|:--|:--|
| [`Maranatha.Utils.QuadratureBoundarySpec._decode_boundary`](@ref) | map a global boundary selector to local left/right endpoint kinds |
| [`Maranatha.Utils.QuadratureBoundarySpec._boundary_at`](@ref) | resolve the scalar boundary symbol used on one axis |
| [`Maranatha.Utils.QuadratureBoundarySpec._validate_boundary_spec`](@ref) | validate a scalar-or-axis-wise boundary specification |

---

## Role in the package

These helpers are reused by:

- quadrature node construction,
- Newton-Cotes admissibility logic,
- derivative-based residual-model selection,
- refinement-family validation,
- TOML parsing and validation,
- report formatting and filename-token generation.

This shared layer avoids repeating boundary parsing rules across unrelated
modules.

---

## Interpretation note

The meaning of a boundary selector depends on the numerical backend:

- Newton-Cotes uses it for open/closed composite tiling behavior,
- Gauss-family rules use it to choose Legendre / Radau / Lobatto variants,
- B-spline quadrature currently accepts only `:LU_ININ` in the public
  node-construction path.

So `QuadratureBoundarySpec` standardizes the selector syntax, while the final numerical
meaning is backend-specific.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.QuadratureBoundarySpec,
]
Private = true
```
