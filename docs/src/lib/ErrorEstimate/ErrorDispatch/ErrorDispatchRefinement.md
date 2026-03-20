# Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement

## Overview

`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement` is the unified rule-family dispatch
layer for the refinement-based error-estimation branch of `Maranatha.jl`.

Where [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative`](@ref) coordinates the
residual-based asymptotic error model, this module provides a much lighter
interface for coarse-versus-refined quadrature comparison.

Its responsibilities are intentionally narrow:

1. identify the quadrature family associated with a given `rule`,
2. dispatch the request to the matching refinement backend,
3. forward shared refinement keywords such as `real_type`, `threaded_subgrid`,
   and optional `I_coarse`,
4. expose a single public entry point for refinement-based error estimation.

This module does **not** construct residual models, does **not** evaluate
high-order derivatives, and does **not** use derivative jets.

---

## Refinement-based error-estimation idea

The refinement branch estimates an error scale by comparing two quadrature
evaluations of the same integrand on the same domain:

- a coarse evaluation using a base subdivision count,
- a refined evaluation using a denser subdivision count.

Conceptually, the estimate is built from

```math
\Delta Q = Q_{\mathrm{fine}} - Q_{\mathrm{coarse}},
```

and the currently used scalar error estimate is the absolute difference

```math
|\Delta Q|.
```

The exact refined subdivision rule depends on the backend:

* Gauss-family rules use a direct doubling convention,
* B-spline rules use a direct doubling convention,
* Newton-Cotes rules may require an adjusted refined subdivision count to remain
  compatible with the composite boundary-tiling constraint.

This makes the refinement branch especially useful when derivative-based probing
is expensive, noisy, or theoretically mismatched to the quadrature
construction.

---

## Architecture

`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement` is intentionally small.

It sits above the rule-family-specific refinement backends:

* [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement`](@ref)
* [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement`](@ref)
* [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement`](@ref)

and below higher-level workflow code such as
[`Maranatha.Runner.run_Maranatha`](@ref).

Its role is simply to provide a uniform public API so that caller-side code does
not need to manually distinguish among Gauss, Newton-Cotes, and B-spline rule
families.

It also centralizes forwarding of an optional precomputed coarse quadrature
value through `I_coarse` when the caller wants to avoid redundant coarse-grid
evaluation inside the selected backend.

---

## Internal rule-family dispatch

The internal helper
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement._dispatch_refinement`](@ref)
performs the actual rule-family selection.

It checks the input `rule` in the following order:

1. Gauss-family rule,
2. Newton-Cotes rule,
3. B-spline rule.

The request is then forwarded to the corresponding backend, together with shared
keywords such as `real_type`, `threaded_subgrid`, and optional `I_coarse`:

* Gauss-family rules →
  [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement.error_estimate_refinement_gauss`](@ref)
* Newton-Cotes rules →
  [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement.error_estimate_refinement_newton_cotes`](@ref)
* B-spline rules →
  [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement.error_estimate_refinement_bspline`](@ref)

If the rule does not belong to any supported refinement family, the dispatch
layer raises a fatal error.

---

## Backend-specific behavior

Although the public interface is unified, the numerical meaning of the refined
evaluation is slightly backend-dependent.

### Gauss-family rules

The Gauss refinement backend compares:

* `Q(N)`
* `Q(2N)`

using the same rule family and boundary handling.

### Newton-Cotes rules

The Newton-Cotes refinement backend also starts from the idea of comparing a
coarse and refined composite rule, but the refined subdivision count must obey
the admissibility structure of the composite Newton-Cotes assembly.

Therefore, the refined candidate `2N` may be adjusted internally to the nearest
valid boundary-compatible subdivision count.

### B-spline rules

The B-spline refinement backend compares:

* `Q(N)`
* `Q(2N)`

while preserving the B-spline kind and, when relevant, the smoothing parameter
`λ`.

For interpolation-type B-spline rules, the effective smoothing parameter is
forced to zero. For smoothing-type rules, the user-supplied `λ` is passed
through after validation.

---

## No derivative or jet layer

Unlike the residual-based dispatch module
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative`](@ref), this refinement dispatch layer
does not coordinate any of the following:

* derivative backend selection,
* scalar derivative caching,
* derivative-jet reuse,
* midpoint residual extraction,
* axis-separable derivative-model assembly.

This is an intentional design choice.

The refinement branch is meant to remain lightweight and operationally simple:
it estimates an error scale from repeated quadrature evaluation rather than from
an asymptotic derivative model.

As a consequence, options such as `use_error_jet` are irrelevant inside this
module and are ignored by higher-level workflow code whenever
`err_method = :refinement` is selected.

---

## Mathematical interpretation note

The refinement estimate produced by this branch should be understood as a
**practical error-scale indicator**, not automatically as a strict error bound
and not necessarily as a fully normalized asymptotic estimator.

At present, the returned scalar estimate is simply based on the magnitude of the
coarse-versus-refined difference recorded by the selected backend.

No additional Richardson-style normalization factor is imposed at this dispatch
layer.

---

## Public API

The main public entry point is:

* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement.error_estimate_refinement`](@ref)

This function is the intended refinement-based counterpart of

* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

in the residual / derivative branch.

The public function is deliberately thin: it forwards the request to the
internal rule-family dispatcher and returns the backend-produced refinement
result object.

If an already computed coarse quadrature value is available, that value can be
passed once through `I_coarse` and will be relayed to the selected backend.

---

## Scope note

This module does **not** define the refinement algorithms themselves.
Those live in the family-specific backends.

It also does **not** perform quadrature directly, except indirectly through the
called backend modules.

Its purpose is organizational:

* unify refinement entry points,
* isolate rule-family selection,
* keep caller-side workflow code simple.

It is therefore best viewed as the orchestration layer of the
refinement-based branch of `Maranatha.ErrorEstimate`.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement,
]
Private = true
```