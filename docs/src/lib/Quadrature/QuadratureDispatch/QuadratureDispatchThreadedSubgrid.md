# Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid

`Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid`
provides the CPU multithreaded subgrid-partitioned tensor-product quadrature
backend of `Maranatha.jl`.

This module evaluates tensor-product quadrature by partitioning the full
quadrature grid into rectangular subblocks across one or more axes, assigning
those blocks to Julia threads, and combining the resulting thread-local partial
sums into a final integral estimate.

Unlike a simple outer-loop threading strategy, this backend is designed to
distribute work more flexibly across dimensions by choosing per-axis split
counts and constructing subgrid blocks from them.

---

## Responsibility in the quadrature layer

Within the quadrature-dispatch system, this module serves as one of the CPU
execution backends:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureNodes` | build 1D quadrature nodes and weights |
| `QuadratureDispatch` | select the appropriate execution backend |
| `QuadratureDispatchThreadedSubgrid` | execute tensor-product quadrature on CPU threads using subgrid partitioning |

This means the module does not define quadrature rules itself. Its role is to
execute the tensor-product accumulation efficiently once the rule and boundary
configuration have already been determined.

---

## Overview

The threaded subgrid backend currently contains five main roles:

| Function | Responsibility |
|:--|:--|
| [`_chunk_range`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid._chunk_range) | split a one-dimensional index range into contiguous chunks |
| [`_choose_axis_splits`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid._choose_axis_splits) | choose a balanced per-axis split configuration for the requested thread budget |
| [`_block_ranges_from_splits`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid._block_ranges_from_splits) | build tensor-product subgrid blocks from the axis splits |
| [`_effective_nthreads_req`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid._effective_nthreads_req) | clamp the requested thread count to the available Julia threads |
| `quadrature_*_threaded_subgrid` | perform dimension-specific threaded tensor-product quadrature |
| [`quadrature_nd_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_nd_threaded_subgrid) | generic fallback for higher-dimensional threaded quadrature |
| [`quadrature_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_threaded_subgrid) | unified public dispatcher |

---

## Specialized and generic paths

This module provides specialized implementations for dimensions 1 through 4:

- [`quadrature_1d_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_1d_threaded_subgrid)
- [`quadrature_2d_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_2d_threaded_subgrid)
- [`quadrature_3d_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_3d_threaded_subgrid)
- [`quadrature_4d_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_4d_threaded_subgrid)

For dimensions above 4, it falls back to the generic
[`quadrature_nd_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_nd_threaded_subgrid) implementation.

This structure keeps the common low-dimensional cases explicit and efficient,
while still preserving a general ND path for broader use.

---

## Notes

- The same one-dimensional quadrature nodes and weights are reused on every
  axis.
- Zero-weight quadrature points are skipped during accumulation.
- The public API is centered on [`quadrature_threaded_subgrid`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid.quadrature_threaded_subgrid).
- Private helpers are included below because this page exposes private autodocs.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchThreadedSubgrid,
]
Private = true
```