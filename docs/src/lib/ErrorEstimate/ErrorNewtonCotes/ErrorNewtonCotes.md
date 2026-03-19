# Maranatha.ErrorEstimate.ErrorNewtonCotes

# Maranatha.ErrorEstimate.ErrorNewtonCotes

Newton-Cotes–specific truncation-error modeling backends for `Maranatha.jl`.

---

## Overview

`ErrorNewtonCotes` provides the error-estimation components specialized for
composite Newton-Cotes quadrature rules inside `Maranatha.ErrorEstimate`.

Because Newton-Cotes rules are built from algebraic interpolation on structured
grids, they admit two complementary error-estimation viewpoints:

- an **exact residual-based viewpoint**, which analyzes the algebraic midpoint
  residual structure of the composite rule itself, and
- a **refinement-based viewpoint**, which measures how the computed quadrature
  value changes when the grid is refined.

This module serves as the Newton-Cotes–specific container for those two
backends.

---

## Supported estimation frameworks

### Residual-based derivative estimation

The derivative-based backend extracts the formal truncation structure of the
composite Newton-Cotes rule.

Its core logic is:

1. construct the composite coefficient vector exactly in rational arithmetic,
2. detect which midpoint-centered residual moments are algebraically nonzero,
3. convert those residuals into Taylor-style coefficients,
4. combine them later with physical derivative probes in the higher-level
   dispatch layer.

Because the residual detection is performed in exact rational arithmetic, this
backend captures the true algebraic structure of the rule without tolerance
ambiguity.

### Refinement-based estimation

The refinement backend provides a more empirical alternative.

Its core logic is:

1. evaluate the quadrature rule at a coarse resolution,
2. evaluate the same rule at a refined resolution,
3. use the difference as a practical error-scale estimate.

This path does not require derivative evaluation or explicit residual analysis
and is useful when practical numerical behavior matters more than the formal
asymptotic structure.

---

## Why Newton-Cotes needs its own backend

Newton-Cotes rules differ from Gauss-family and B-spline rules in several
important ways:

- the nodes are tied to a structured interpolation grid,
- the composite rule can be assembled exactly in rational arithmetic,
- the formal residual structure is strongly linked to polynomial exactness,
- high-order rules may exhibit large coefficients and practical stability issues.

As a result, Newton-Cotes rules benefit from both:

- an **exact structural residual analysis**, and
- a **numerical refinement check**.

The two views are complementary rather than redundant.

---

## Module structure

`ErrorNewtonCotes` contains two main specialized submodules:

- [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative`](@ref)  
  Exact-rational residual extraction for derivative-based truncation-error
  modeling.

- [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement`](@ref)  
  Refinement-difference estimators based on coarse-versus-refined quadrature
  comparison.

Unified selection between these strategies is handled by
[`Maranatha.ErrorEstimate.ErrorDispatch`](@ref).

---

## Role in the package

This module is used internally by the global error-estimation layer when a
Newton-Cotes quadrature rule is selected.

In a typical workflow, users do not call these backends directly. Instead, they
invoke higher-level entry points such as:

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)
- [`Maranatha.Runner.run_Maranatha`](@ref)

which then route to the appropriate Newton-Cotes backend automatically.

---

## Notes

- These backends provide **error scales**, not rigorous bounds.
- The derivative-based and refinement-based estimators may differ in both cost
  and interpretation.
- Exact residual structure reflects the formal algebraic behavior of the rule,
  while refinement reflects the observed numerical behavior under resolution
  change.
- High-order Newton-Cotes rules can become numerically delicate even when their
  formal residual structure is known exactly.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorNewtonCotes,
]
Private = true
```