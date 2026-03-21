# Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement

## Overview

`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement` provides the refinement-based
error-estimation backend for B-spline quadrature rules inside
`Maranatha.ErrorEstimate`.

Instead of analyzing midpoint residual moments, this module estimates the
integration error by comparing quadrature results computed at two different
resolutions.

Like the main B-spline residual backend, computations are performed in
floating-point arithmetic.

When an already computed coarse quadrature value is available, the refinement
backend can also reuse it through `I_coarse` instead of recomputing the coarse
grid evaluation.

---

## Refinement model

Let

```math
I_h
```

be the quadrature result using subdivision size `h`, and

```math
I_{h/2}
```

be the result obtained from a refined grid.

The refinement error indicator is based on their difference:

```math
\Delta I = I_{h/2} - I_h .
```

Under the standard asymptotic convergence assumption,

```math
I_h = I^\* + C h^p + O(h^{p+1}),
```

this difference behaves as

```math
\Delta I \approx C (2^{-p} - 1) h^p ,
```

which makes `|ΔI|` a practical proxy for the truncation error magnitude.

This approach avoids explicit derivative evaluation and residual modeling.

---

## Grid refinement strategy

For B-spline rules, refinement is performed by constructing a new spline
quadrature with a finer subdivision count.

Unlike Newton-Cotes rules, B-spline quadrature does not impose strict tiling
constraints on admissible subdivision counts. Therefore, refinement typically
uses a simple proportional rule such as doubling the subdivision count.

The refined rule is generated using the same spline family and smoothing
parameters as the original rule so that differences reflect resolution changes
only.

For axis-wise `rule` specifications, the refinement dispatch layer currently
accepts the request only when all active axes remain within the B-spline
family.

---

## Supported spline families

This backend supports the same spline rule families as the main B-spline module:

* interpolation splines: `:bspline_interp_p2`, `:bspline_interp_p3`, ...
* smoothing splines: `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

For smoothing splines, the smoothing parameter `\lambda` is preserved when
constructing the refined rule.

In the current public quadrature path, B-spline node construction still
requires `boundary = :LU_ININ`.

---

## Why refinement is attractive for splines

B-spline quadrature rules are typically smooth and stable under grid refinement.
As a result:

* refinement differences are well-behaved,
* no residual scanning is required,
* no derivative backends are involved,
* the method can be significantly faster than derivative-based estimators.

This makes refinement-based estimation a strong and robust choice for spline
rules in practical workloads.

---

## Function roles

### Refined quadrature construction

The backend obtains the coarse quadrature value at subdivision count `N`,
either by computing it internally or by reusing a caller-supplied `I_coarse`,
and rebuilds the spline quadrature on a refined grid while preserving:

* rule type,
* spline degree,
* smoothing parameter (if present),
* boundary configuration.

### Error indicator computation

The main output is a scalar error indicator derived from the difference between
coarse and refined quadrature values.

When `I_coarse` is supplied, only the refined-grid quadrature value needs to be
newly evaluated inside this backend.

---

## Scope note

This backend provides only refinement-based error indicators.

It does not:

* analyze residual moment structure,
* compute derivative probes,
* assemble multidimensional axis-separated models,
* provide rigorous bounds.

Its output is intended for use by higher-level orchestration code that selects
between error-estimation strategies.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement,
]
Private = true
```
