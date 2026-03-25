# Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative

## Overview

`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative` is the coordination layer of the
residual-based error-model stack.

It sits between the rule-specific residual backends and the dimension-specific
error estimators, and it also exposes the public entry points used elsewhere in
`Maranatha.jl`.

This module specifically covers the **residual-based / derivative-based**
branch of the error-estimation framework. The complementary refinement-based
branch is coordinated separately by
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement`](@ref).

Its responsibilities fall into three broad categories:

1. normalize residual-term extraction across Newton-Cotes, Gauss, and B-spline rules,
2. provide a unified derivative-backend interface,
3. dispatch to specialized or generic multidimensional error estimators.

---

## Residual-term normalization

The helper [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative._leading_residual_terms_any`](@ref) converts the currently
supported rule families into one common return format:

- `ks`: detected residual orders,
- `coeffs_float`: factorial-scaled residual coefficients in `Float64`,
- `center`: current center tag, presently `:mid`.

This lets the downstream estimators remain agnostic about whether the residual
data came from:

- exact rational Newton-Cotes logic,
- floating-point Gauss logic,
- floating-point B-spline logic.

For repeated use with the same rule configuration, the companion helper
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative._get_residual_model_fixed`](@ref)
stores and reuses the normalized residual-term model through an internal cache.
This avoids rebuilding the same residual metadata across multiple estimator calls.

---

## Residual-model implementation note

Rule-family-specific residual logic is handled inside the underlying residual
backends and normalized by
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative._leading_residual_terms_any`](@ref).

The dispatch layer itself is intentionally kept lightweight: it does not expose
separate public logic for every special-case rule family, but instead presents a
uniform residual-term interface to the downstream estimators.

---

## Derivative backend interface

The derivative wrapper [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.nth_derivative`](@ref) unifies the differentiation
methods used by the error estimators.

Supported backends are:

- `:forwarddiff`
- `:taylorseries`
- ``
- `:enzyme`

The scalar wrapper first checks an internal derivative cache and then dispatches
to the selected backend. If the selector is invalid, it emits a context-rich
fatal error.


The module also exposes
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!`](@ref),
which clears the derivative and residual-model caches at the start of a fresh
run when desired.

These caches and derivative helpers are specific to the residual-based branch
and are not used by the refinement-based path exposed through
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement`](@ref).

### Included backend helpers

The backend-specific derivative helpers include:

- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADForwardDiff.nth_derivative_forwarddiff`](@ref)
- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADTaylorSeries.nth_derivative_taylor`](@ref)
- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADFastDifferentiation.nth_derivative_fastdifferentiation`](@ref)
- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADEnzyme.nth_derivative_enzyme`](@ref).

Each backend has slightly different strengths:

- [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) is the default practical path,
- [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/) is useful for single-pass higher-order expansion,
- [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/) is present mainly as an experimental / benchmarking path here.

---

## Dimension-specific estimators

The included estimator files implement two layers:

### Specialized estimators for ``d = 1,2,3,4``

- [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct_1d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct_2d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct_3d`](@ref)
- [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct_4d`](@ref)

These versions use explicit loop structures and are meant to keep the common
low-dimensional cases transparent and easy to inspect.

### Generic ``n``-dimensional estimators

- [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct_nd`](@ref)

This provides the same axis-separable model for arbitrary dimension using a
generic odometer-style multi-index traversal, with the `*_jet` path reusing
multiple derivative orders through shared derivative-jet evaluations.

---

## Common mathematical structure

All error estimators use the same conceptual model:

```math
E \approx \sum_{i=1}^{n_{\text{err}}}
\texttt{coeff}_{k_i} \, h^{k_i+1} \,
\sum_{\mu=1}^{\texttt{dim}} I_\mu^{(k_i)},
```

where each `I_μ^{(k_i)}` is a cross-axis integral containing a `k_i`-th derivative
along one selected axis and midpoint insertion along that axis,

```math
I_{\mu}^{(k_i)} =
\int\limits_a^b \cdots \int\limits_a^b
\left( \prod_{\nu \neq \mu} dx_{\nu} \right)
\; \frac{\partial^{k_i} f}{\partial x_{\mu}^{k_i}}
\left( x_1, \ldots, x_{\mu}=\bar{x}, \ldots, x_{\texttt{dim}} \right) \,.
```

This means the model is always **axis-separable**. Mixed derivatives are not
the main target of this layer and are intentionally omitted as higher-order
effects.

---

## Public API

Primary public entry points are:

* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

Both are thin dispatchers that select the appropriate dimension-specific
implementation. The `*_jet` variant reuses derivative jets internally, while
the direct variant evaluates scalar derivatives independently.

For refinement-based error estimation, see the separate dispatch layer
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement`](@ref), whose public entry point
is [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement.error_estimate_refinement`](@ref).

---

## Scope note

This module does **not** define quadrature rules or residual backends itself.
Instead, it coordinates:

* residual extraction,
* derivative evaluation,
* multidimensional assembly of the axis-separated model.

It is therefore best thought of as the orchestration layer of the
residual-based branch of `Maranatha.ErrorEstimate`, not as a standalone
numerical method in isolation.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative,
]
Private = true
```