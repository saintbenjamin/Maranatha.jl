# Maranatha.Quadrature.QuadratureDispatch

## Overview

`Maranatha.Quadrature.QuadratureDispatch` is the rule-dispatch layer that connects
the backend quadrature engines to the tensor-product integration interface.

It serves two main purposes:

1. convert a user-facing `(rule, boundary)` selection into concrete `1`-dimensional
   nodes and weights,
2. evaluate tensor-product quadrature sums in dimensions `1`, `2`, `3`, `4`, or
   general `dim`.

In practice, this module is the point where the backend rule families become a
single uniform integration API.

---

## Rule dispatch policy

The public node/weight entry point is:

- [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref)

It dispatches by rule family:

### Newton-Cotes family

Rules of the form:

- `:newton_p3`, `:newton_p4`, `:newton_p5`, ...

are delegated to [`Maranatha.Quadrature.NewtonCotes`](@ref).

The dispatch flow is:

1. parse the local node count `p`,
2. validate / decode the boundary mode,
3. retrieve the composite global coefficient vector ``\beta``,
4. generate uniform nodes on ``[a,b]``,
5. convert coefficients into physical weights via ``w_j = \beta_j \, h``.

This keeps the exact-rational assembly isolated in the Newton-Cotes backend.

### Gauss family

Rules of the form:

- `:gauss_p2`, `:gauss_p3`, `:gauss_p4`, ...

are delegated to [`Maranatha.Quadrature.Gauss`](@ref).

That backend constructs composite Gauss-family nodes and weights by repeating a
single-block Gauss rule across ``N`` uniform subintervals, with endpoint-sensitive
variants applied only where the global boundary actually touches the interval edge.

### B-spline family

Rules of the form:

- `:bspline_interp_p2`, `:bspline_interp_p3`, ...
- `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

are delegated to [`Maranatha.Quadrature.BSpline`](@ref).

At present, this dispatch layer enforces the policy that B-spline quadrature is
supported only for `boundary = :LU_ININ`, i.e. the clamped case.

---

## Boundary decoding

The helper [`Maranatha.Quadrature.QuadratureDispatch._decode_boundary`](@ref) translates the global boundary selector into
local endpoint tags for Newton-Cotes assembly:

- `:LU_ININ` -> `(:closed, :closed)`
- `:LU_EXIN` -> `(:opened, :closed)`
- `:LU_INEX` -> `(:closed, :opened)`
- `:LU_EXEX` -> `(:opened, :opened)`

This helper is intentionally small, but it plays an important role in keeping the
boundary convention centralized and consistent across the stack.

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

All tensor-product drivers integrate over a hypercube with the **same**
interval applied on every axis:

```math
[a,b]^{\texttt{dim}}.
```

This module does **not** implement mixed bounds such as
$[a_1,b_1] \times [a_2,b_2] \times \cdots$.
That restriction is deliberate and keeps the interface aligned with the rest of
the package.

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

This module is intentionally a dispatch-and-accumulation layer only.

It does **not**:

- derive quadrature rules itself,
- implement adaptive sampling,
- perform error estimation,
- apply multithreading,
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