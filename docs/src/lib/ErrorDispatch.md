# Maranatha.ErrorEstimate.ErrorDispatch

## Overview

`Maranatha.ErrorEstimate.ErrorDispatch` is the coordination layer of the
residual-based error-model stack.

It sits between the rule-specific residual backends and the dimension-specific
error estimators, and it also exposes the public entry points used elsewhere in
`Maranatha.jl`.

Its responsibilities fall into three broad categories:

1. normalize residual-term extraction across Newton-Cotes, Gauss, and B-spline rules,
2. provide a unified derivative-backend interface,
3. dispatch to specialized or generic multidimensional error estimators.

---

## Residual-term normalization

The helper [`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_terms_any`](@ref) converts the currently
supported rule families into one common return format:

- `ks`: detected residual orders,
- `coeffs_float`: factorial-scaled residual coefficients in `Float64`,
- `center`: current center tag, presently `:mid`.

This lets the downstream estimators remain agnostic about whether the residual
data came from:

- exact rational Newton-Cotes logic,
- floating-point Gauss logic,
- floating-point B-spline logic.

The companion helper [`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any`](@ref) provides a
lighter-weight path when only the residual indices are needed.

---

## Special handling for Gauss-Radau

Inside [`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any`](@ref), Gauss-Radau rules use a
special branch that bypasses the generic floating-point moment scan.

The reason is practical: in the Radau case, the leading residual sequence is
known analytically, while a naive tolerance-based scan can misclassify very
low-order moments due to floating cancellation.

This branch therefore returns the expected sequence directly, preserving the
stability of the downstream extrapolation/error model.

---

## Derivative backend interface

The derivative wrapper [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative`](@ref) unifies the differentiation
methods used by the error estimators.

Supported backends are:

- `:forwarddiff`
- `:taylorseries`
- `:fastdifferentiation`
- `:enzyme`

The wrapper itself is intentionally simple: it dispatches to the selected
backend and emits a context-rich fatal error if the selector is invalid.

### Included backend helpers

The included `nth_derivative.jl` file defines the backend-specific helpers:

- [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative_forwarddiff`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative_taylor`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative_fastdifferentiation`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative_enzyme`](@ref)

Each backend has slightly different strengths:

- [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) is the default practical path,
- [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/) is useful for single-pass higher-order expansion,
- [`FastDifferentiation.jl`](https://brianguenter.github.io/FastDifferentiation.jl/stable/) provides symbolic differentiation when the integrand is compatible,
- [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/) is present mainly as an experimental / benchmarking path here.

---

## Dimension-specific estimators

The included estimator files implement two layers:

### Specialized estimators for `1D`–`4D`

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_1d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_2d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_3d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_4d`](@ref)

and their threaded companions:

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_1d_threads`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_2d_threads`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_3d_threads`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_4d_threads`](@ref)

These versions use explicit loop structures and are meant to keep the common
low-dimensional cases transparent and easy to inspect.

### Generic `nd` estimators

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_nd`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_nd_threads`](@ref)

These provide the same axis-separable model for arbitrary dimension using a
generic odometer-style multi-index traversal.

---

## Common mathematical structure

All error estimators use the same conceptual model:

```math
E \approx \sum_{i=1}^{n_{\text{err}}}
\texttt{coeff}_{k_i} h^{k_i+1}
\sum_{\mu=1}^{\texttt{dim}} I_\mu^{(k_i)},
```

where each `I_μ^(k)` is a cross-axis integral containing a `k`-th derivative
along one selected axis and midpoint insertion along that axis.

This means the model is always **axis-separable**. Mixed derivatives are not
the main target of this layer and are intentionally omitted as higher-order
effects.

---

## Threading strategy

The threaded estimators keep the same mathematical definition as the non-threaded
versions, but parallelize the dominant summation loops.

Typical policy:

- distribute independent node or axis loops via `Threads.@threads`,
- accumulate partial sums into thread-local buffers,
- reduce those partial sums at the end.

The exact parallelization pattern differs by dimension:

- in `1D`, threading is over residual-term loops,
- in `2D`, `3D`, and `4D`, threading is over flattened tensor-product index grids,
- in generic `nd`, threading is over axis contributions.

This design favors simple thread safety and deterministic local work partitioning
rather than aggressive optimization.

---

## Public API

The two main public entry points are:

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_threads`](@ref)

These are thin dispatchers that select the dimension-specific implementation.

They intentionally do not duplicate the full mathematical logic themselves; that
logic lives in the included estimator files.

---

## Scope note

This module does **not** define quadrature rules or residual backends itself.
Instead, it coordinates:

- residual extraction,
- derivative evaluation,
- multidimensional assembly of the axis-separated model.

It is therefore best thought of as the orchestration layer of
`Maranatha.ErrorEstimate`, not as a standalone numerical method in isolation.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorDispatch,
]
Private = true
```