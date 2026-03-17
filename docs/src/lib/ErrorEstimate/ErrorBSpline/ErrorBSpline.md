# Maranatha.ErrorEstimate.ErrorBSpline

Truncation-error modeling backends for B-spline quadrature rules in
`Maranatha.jl`.

---

## Overview

`ErrorBSpline` provides error-estimation tools tailored to B-spline–based
quadrature schemes. It supports both residual-based asymptotic modeling and
refinement-based practical error estimation.

B-spline rules differ from classical polynomial rules in that they are built
from compactly supported basis functions with strong locality properties.
This structure affects both the residual spectrum and the behavior under
grid refinement, requiring dedicated error models.

---

## Supported frameworks

Two complementary approaches are implemented:

### Residual-based model

Analyzes the intrinsic truncation structure of the composite rule.

Key steps:

1. Construct the dimensionless composite rule on a tiled grid.
2. Detect leading nonzero residual moment orders specific to the B-spline rule.
3. Combine those coefficients with midpoint derivative probes.
4. Produce an asymptotic truncation-error scale model.

This approach captures theoretical convergence behavior and can be useful for
stabilizing extrapolation fits.

---

### Refinement-based model

Uses direct comparison between coarse and refined evaluations.

Key steps:

1. Evaluate the quadrature rule on a coarse subdivision count.
2. Recompute using a refined subdivision count.
3. Use the difference as a practical error estimate.

This method avoids derivative evaluation and is often more robust for complex
integrands or when high-order derivatives are unavailable or expensive.

---

## Characteristics of B-spline rules

B-spline quadrature exhibits:

- strong locality due to compact support,
- smoothness determined by spline degree,
- nontrivial residual patterns compared to polynomial rules,
- favorable stability under refinement.

Error behavior may therefore differ from Newton–Cotes or Gauss families,
especially at low resolutions.

---

## Module structure

`ErrorBSpline` serves as a container that re-exports specialized submodules:

- `ErrorBSplineDerivative` — residual-based estimators
- `ErrorBSplineRefinement` — refinement-difference estimators

Unified dispatch across rule families is handled by
`Maranatha.ErrorEstimate.ErrorDispatch`.

---

## Role in the package

This module is used internally by the error-estimation layer and is typically
accessed through high-level interfaces such as:

- `error_estimate_derivative_direct`
- `error_estimate_derivative_jet`
- `error_estimate_refinement`

Users normally do not call the low-level B-spline backends directly.

---

## Notes

- These estimators provide error scales, not strict bounds.
- Performance and stability depend on spline degree, resolution, and integrand smoothness.
- Refinement-based estimation can be especially effective for spline rules due to their locality.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorBSpline,
]
Private = true
```