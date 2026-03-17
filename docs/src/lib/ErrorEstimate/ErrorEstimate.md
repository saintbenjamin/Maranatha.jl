# Maranatha.ErrorEstimate

## Overview

`Maranatha.ErrorEstimate` provides the truncation-error modeling layer for the
structured quadrature rules used in `Maranatha.jl`.

It implements two complementary estimation frameworks:

- a residual-based asymptotic truncation-error model driven by midpoint
  residual structure and derivative probes, and
- a refinement-based model that estimates error empirically from
  coarse-versus-refined quadrature evaluations.

These tools are intended for:

- fit stabilization,
- expected error-scaling diagnostics,
- automatic detection of leading asymptotic behavior,
- practical error-scale estimation when derivative probes are undesirable.

In the overall workflow, this module sits between quadrature construction and
least-χ² fitting.

---

## Core modeling ideas

### Residual-based branch

The residual branch constructs an asymptotic model from rule structure:

1. build the composite quadrature rule on a dimensionless tiling grid,
2. identify leading nonzero midpoint-centered residual moments,
3. convert those residuals into factorial-scaled coefficients,
4. combine the coefficients with midpoint derivative probes of the integrand.

Conceptually, the modeled error takes the form

```math
E \approx \sum_{i=1}^{n_{\texttt{err}}}
\texttt{coeff}_{k_i} \, h^{k_i+1} \,
\sum_{\mu=1}^{\texttt{dim}} I_\mu^{(k_i)},
```

where each `I_\mu^{(k)}` corresponds to a cross-axis integral containing a
single-axis derivative of order `k`.

This construction is intentionally **axis-separable**; mixed-derivative terms
are treated as higher-order corrections and are not part of the primary model.

### Refinement-based branch

The refinement branch estimates an error scale by comparing quadrature results
computed at different resolutions:

1. evaluate the rule at a coarse subdivision count,
2. evaluate the same rule on a refined grid,
3. use the difference as a practical error indicator.

This approach avoids derivative evaluation and residual modeling entirely and
is particularly useful when derivatives are expensive, unstable, or conceptually
mismatched to the quadrature construction.

---

## Residual backends

Residual extraction is handled by rule-family-specific modules:

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative`](@ref)

Exact-rational residual detection for Newton-Cotes rules.

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative`](@ref)

Tolerance-based residual detection in `Float64` for Gauss-family rules.

### [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative`](@ref)

Tolerance-based residual detection for B-spline quadrature rules.

### [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative`](@ref)

Unified coordination layer for residual extraction, derivative probing, and
dimension-specific or generic estimators.

---

## Refinement backends

Refinement-difference estimation is implemented separately:

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement`](@ref)

Refinement estimator for Newton-Cotes rules.

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement`](@ref)

Refinement estimator for Gauss-family rules.

### [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement`](@ref)

Refinement estimator for B-spline rules.

### [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement`](@ref)

Unified dispatcher for refinement-based error estimation.

---

## Derivative backends

High-order derivatives in the residual branch are evaluated through

[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.nth_derivative`](@ref).

Supported backends include:

* `:forwarddiff`
* `:taylorseries`
* `:fastdifferentiation`
* `:enzyme`

Derivative jets are handled separately by the jet module.

---

## Centering convention

Residuals are currently defined using a midpoint-centered convention.
The dispatch layer returns the center tag explicitly so that downstream code
remains agnostic to future centering policies.

---

## Public API

Primary entry points include:

* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)
* [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement.error_estimate_refinement`](@ref)

These interfaces support dimension-specific implementations for low dimensions
and generic multidimensional estimators for higher dimensions.

---

## Design scope

This module focuses strictly on truncation-error modeling.

It does **not**:

* perform least-$\\chi^2$ fitting,
* define quadrature rules,
* provide rigorous interval bounds,
* hide tolerance choices inside unrelated layers.

---

## Practical note

Floating-point residual detection depends on backend tolerances for Gauss and
B-spline rules, while Newton-Cotes residual detection uses exact rational
arithmetic.

Refinement-based estimators reflect the actual numerical convergence behavior
but do not expose formal asymptotic coefficients.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate,
]
Private = true
```