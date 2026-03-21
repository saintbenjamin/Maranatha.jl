# Maranatha.Quadrature

`Maranatha.Quadrature` is the deterministic rule-dispatched integration layer
of `Maranatha.jl`.

It provides the package's tensor-product quadrature engine together with the
rule-family-specific helpers needed to construct 1D nodes and weights.

---

## Overview

The quadrature layer currently supports three rule families:

- Newton-Cotes
- Gauss-family rules
- B-spline rules

and extends them to multidimensional tensor-product integration over:

- scalar hypercube domains `[a,b]^dim`, and
- axis-wise rectangular domains specified by tuples or vectors of endpoints.

Both `rule` and `boundary` may be passed either as:

- one scalar symbol shared across all axes, or
- a tuple / vector with one entry per axis.

---

## Main layers

The quadrature stack is organized into several cooperating modules.

| Layer | Responsibility |
|:--|:--|
| [`Maranatha.Quadrature.NewtonCotes`](@ref) | exact-rational Newton-Cotes parsing and weight assembly |
| [`Maranatha.Quadrature.Gauss`](@ref) | Gauss / Radau / Lobatto rule construction |
| [`Maranatha.Quadrature.BSpline`](@ref) | interpolation and smoothing B-spline node/weight construction |
| [`Maranatha.Quadrature.QuadratureRuleSpec`](@ref) | scalar-versus-axis-wise rule validation and normalization |
| [`Maranatha.Utils.QuadratureBoundarySpec`](@ref) | scalar-versus-axis-wise boundary validation and endpoint decoding |
| [`Maranatha.Quadrature.QuadratureNodes`](@ref) | 1D node/weight construction |
| [`Maranatha.Quadrature.QuadratureDispatch`](@ref) | tensor-product accumulation and execution-backend selection |

---

## Supported rule families

### Newton-Cotes

Composite Newton-Cotes rules are assembled from exact rational data before
conversion to floating-point weights.

Typical symbols:

```julia
:newton_p3, :newton_p4, :newton_p5, ...
```

Boundary selectors determine the open/closed endpoint treatment of the
composite assembly.

### Gauss-family rules

The Gauss backend supports Legendre, Radau, and Lobatto variants through the
common `:gauss_p*` rule symbols together with the boundary selector.

Typical symbols:

```julia
:gauss_p2, :gauss_p3, :gauss_p4, ...
```

### B-spline rules

The B-spline backend supports interpolation and smoothing variants.

Typical symbols:

```julia
:bspline_interp_p2, :bspline_interp_p3, ...
:bspline_smooth_p2, :bspline_smooth_p3, ...
```

In the current public node-construction path, B-spline quadrature accepts only
`boundary = :LU_ININ`.

---

## Multidimensional strategy

All multidimensional integration is performed by tensor-product accumulation:

```math
\sum_{i_1,\ldots,i_d} w^{(1)}_{i_1} \cdots w^{(d)}_{i_d}
f(x^{(1)}_{i_1}, \ldots, x^{(d)}_{i_d}).
```

Important implementation details:

- 1D node and weight sets are constructed independently on each active axis,
- axis-wise domains, rules, and boundaries are supported,
- zero-weight entries may be skipped,
- accumulation order is deterministic,
- execution may use the plain CPU path, the threaded-subgrid path, or CUDA,
  depending on the selected backend.

The common low-dimensional entry points are:

- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_1d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_2d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_3d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_4d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_nd`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref)

---

## Boundary semantics

The shared boundary symbols

```julia
:LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX
```

have backend-dependent numerical meaning:

- Newton-Cotes uses them for composite open/closed endpoint treatment,
- Gauss-family rules use them to select Legendre / Radau / Lobatto variants,
- B-spline rules currently use only `:LU_ININ` in the public quadrature path.

The selector syntax is unified, but its interpretation is family-specific.

---

## Scope notes

`Maranatha.Quadrature` is intentionally deterministic and explicit.

It does **not** provide:

- adaptive quadrature,
- hidden rule mutation,
- opaque black-box refinement heuristics.

Instead, it emphasizes:

- explicit rule selection,
- transparent tensor-product structure,
- reproducible accumulation order,
- family-specific validation of admissible configurations.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature,
]
Private = true
```
