# Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA

`Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA` provides the
CUDA-based tensor-product quadrature backend of `Maranatha.jl`.

This module is responsible for evaluating multi-dimensional quadrature sums on
the GPU. It constructs one-dimensional quadrature nodes and weights through the
shared quadrature-node layer, transfers them to CUDA device memory, launches a
CUDA kernel over the tensor-product grid, and reduces the resulting partial
contributions into a final scalar quadrature value.

---

## Responsibility in the quadrature layer

Within the quadrature-dispatch system, this module serves as the GPU execution
backend:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureNodes` | build 1D quadrature nodes and weights |
| `QuadratureDispatch` | select the appropriate execution backend |
| `QuadratureDispatchCUDA` | execute tensor-product quadrature on CUDA devices |

In other words, this module does not define the quadrature rules themselves.
Instead, it provides a CUDA implementation of the tensor-product accumulation
step once the node/weight rule has already been chosen.

---

## Overview

The CUDA backend currently consists of four main internal roles:

| Function | Responsibility |
|:--|:--|
| [`_linear_to_indices_cuda`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA._linear_to_indices_cuda) | convert a flattened tensor-product index into per-axis node indices |
| [`_weight_product_cuda`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA._weight_product_cuda) | compute the tensor-product weight for a given multi-index |
| [`_eval_f_cuda`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA._eval_f_cuda) | evaluate the integrand at the selected tensor-product node |
| [`_kernel_quadrature_nd!`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA._kernel_quadrature_nd!) | execute the grid-stride CUDA quadrature kernel |
| [`quadrature_cuda`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA.quadrature_cuda) | public entry point for CUDA quadrature evaluation |

---

## Notes

- This module assumes that the supplied integrand is CUDA-compatible.
- The same one-dimensional quadrature nodes and weights are reused on every
  axis.
- The public API is centered on [`quadrature_cuda`](@ref Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA.quadrature_cuda).
- The helper functions documented below are internal implementation details, but
  they are included here because this page exposes private autodocs.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureDispatch.QuadratureDispatchCUDA,
]
Private = true
```