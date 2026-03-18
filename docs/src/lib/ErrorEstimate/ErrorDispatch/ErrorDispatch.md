# Maranatha.ErrorEstimate.ErrorDispatch

Unified dispatch layer for truncation-error estimation in `Maranatha.jl`.

---

## Overview

`ErrorDispatch` provides the high-level routing logic that connects quadrature
rules, derivative backends, and refinement strategies to the appropriate
error-estimation implementation.

Rather than exposing rule-specific estimators directly, this module offers a
stable interface that automatically selects the correct backend based on:

- quadrature rule family,
- boundary configuration,
- requested estimation method,
- derivative backend policy,
- problem dimension.

This design isolates user-facing workflows from low-level implementation
details.

---

## Supported estimation paths

Two major frameworks are supported.

### Residual-based (derivative) estimation

Uses structural properties of the composite quadrature rule together with
high-order derivative probes.

Typical workflow:

1. Identify the rule family and boundary pattern.
2. Retrieve or build the corresponding residual model.
3. Compute midpoint derivatives using the selected backend.
4. Assemble a truncation-error scale estimate.

Derivative evaluation may be performed using direct or jet-based methods.

---

### Refinement-based estimation

Uses direct comparison between coarse and refined quadrature evaluations.

Typical workflow:

1. Evaluate the rule at subdivision count `N`.
2. Re-evaluate at a refined count (e.g., `2N`).
3. Use the difference as a practical error estimate.

This path avoids derivative computation entirely and is often preferred for
expensive or non-smooth integrands.

---

## Dispatch responsibilities

`ErrorDispatch` is responsible for:

- selecting the correct rule-family backend  
  (Newton–Cotes, Gauss, B-spline, etc.),
- coordinating derivative evaluation policies,
- handling dimension-specific versus generic implementations,
- managing shared caches used by derivative-based estimators,
- providing a consistent return format across methods.

---

## Public entry points

The primary public interface is:

- `error_estimate`

Additional functions such as
`error_estimate_derivative_direct`,
`error_estimate_derivative_jet`,
and `error_estimate_refinement`
serve as backend implementations and are typically not called directly.

---

## Caching behavior

Residual-based estimation may reuse cached data, including:

- residual model coefficients,
- individual derivative values,
- derivative jets.

Caches can be cleared via dedicated utility functions when a fresh evaluation
is required.

The refinement-based path generally does not rely on derivative caches.

---

## Role in the package

`ErrorDispatch` forms the bridge between:

- quadrature evaluation (`QuadratureDispatch`),
- derivative computation (`AutoDerivative`),
- rule-family-specific error models,
- high-level convergence workflows.

Users normally interact with this module indirectly through runner or fitting
tools.

---

## Notes

- Estimates represent error scales, not guaranteed bounds.
- Computational cost and stability depend on rule type, dimension, and
  integrand behavior.
- Different backends may produce different asymptotic interpretations of the
  truncation error.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorDispatch,
]
Private = true
```