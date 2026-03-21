# Maranatha.Quadrature.QuadratureDispatch

## Overview

`Maranatha.Quadrature.QuadratureDispatch` is the tensor-product evaluation
layer that connects backend quadrature rules to a uniform integration interface.

Its primary role is to evaluate tensor-product quadrature sums in dimensions
`1`, `2`, `3`, `4`, or general `dim`, using nodes and weights supplied by
the quadrature-node generator.

In practice, this module provides the common integration API used by the
higher-level workflow once rule-specific nodes and weights are available.

---

## Rule-family handling

Rule-family interpretation and node/weight construction are performed by the
quadrature-node generator:

- [`Maranatha.Quadrature.QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)

This module treats the generated nodes and weights as opaque inputs and does
not depend on rule-specific implementation details.

Consequently, all rule-family logic (Newton-Cotes, Gauss, B-spline, etc.)
is encapsulated outside the dispatch layer.

---

## Boundary handling

The helper [`Maranatha.Utils.QuadratureBoundarySpec._decode_boundary`](@ref)
translates the global boundary selector into local endpoint tags used by
rule-specific backends.

This dispatch layer does not interpret boundary semantics directly, but
passes the selector to the underlying node generator, which relies on the
dedicated `QuadratureBoundarySpec` module.

---

## Tensor-product strategy

After ``1``-dimensional nodes and weights are constructed, this module evaluates the
full quadrature by explicit tensor-product accumulation.

### Specialized low-dimensional paths

The functions

- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_1d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_2d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_3d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_4d`](@ref)

use explicit nested loops. This keeps the common low-dimensional cases simple and
fully transparent.

### General `dim` path

For dimensions higher than `4`, the function [`Maranatha.Quadrature.QuadratureDispatch.quadrature_nd`](@ref) uses an
odometer-style multi-index update.

The algorithm:

1. allocate an integer index vector of length `dim`,
2. allocate an argument buffer of length `dim`,
3. at each step:
   - load coordinates from `xs`,
   - multiply the corresponding weights,
   - call `f(args...)`,
4. increment the multi-index lexicographically until exhaustion.

This preserves deterministic loop ordering while avoiding recursive machinery.

---

## Deterministic accumulation

Throughout this module:

- the accumulation order is explicit,
- zero weights are skipped early,
- no adaptive refinement is introduced,
- no parallel reduction is performed.

This matches the overall design philosophy of `Maranatha.jl`: predictable,
reproducible numerical behavior with minimal hidden control flow.

---

## Domain convention

Tensor-product drivers support both uniform and axis-specific bounds.

- If scalar endpoints are provided, the same interval is applied to all axes:

```math
[a,b]^{\texttt{dim}}.
```

* If tuples or vectors are provided, each axis may use distinct bounds:

```math
[a_1,b_1] \times [a_2,b_2] \times \cdots \times [a_{\texttt{dim}}, b_{\texttt{dim}}].
```

The dispatch layer treats the domain as an external specification and simply
uses the coordinates supplied to the quadrature-node generator.


---

## Complexity note

If the ``1``-dimensional rule has `length(xs)` nodes, then the tensor-product
cost scales as:

```math
\mathcal{O}(\texttt{length(xs)}^{\texttt{dim}}).
```

This is appropriate for structured deterministic quadrature experiments, but it
becomes expensive quickly as `dim` grows.

---

## Scope notes

This module is intentionally a tensor-product accumulation layer only.

It does **not**:

- construct quadrature nodes or weights,
- derive quadrature rules,
- implement adaptive sampling,
- perform error estimation,
- apply multithreading (except via optional sub-dispatch backends),
- change rule semantics based on the integrand.

All such behavior belongs in other parts of the `Maranatha.jl` stack.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureDispatch,
]
Private = true
```
