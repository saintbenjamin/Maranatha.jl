# Maranatha.ErrorEstimate

## Overview

`Maranatha.ErrorEstimate` provides the residual-based truncation-error modeling
layer for the structured quadrature rules used in `Maranatha.jl`.

Its role is not to produce rigorous bounds, but to build a consistent asymptotic
error model that can be used for:

- fit stabilization,
- expected error-scaling diagnostics,
- leading-order detection from residual structure.

In the overall workflow, this module sits between quadrature construction and
least-$\chi^2$ fitting.

---

## Core modeling idea

The module follows a common pattern across all supported rule families:

1. build the composite quadrature rule on a dimensionless tiling grid,
2. identify the first nonzero midpoint-centered residual moments,
3. convert those residual moments into factorial-scaled coefficients,
4. combine the coefficients with midpoint-based derivative probes of the integrand.

Conceptually, the modeled error takes the form

```math
E \approx \sum_{i=1}^{n_{\texttt{err}}}
\texttt{coeff}_{k_i} \, h^{k_i+1} \,
\sum_{\mu=1}^{\texttt{dim}} I_\mu^{(k_i)},
```

where each ``I_\mu^{(k)}`` corresponds to a cross-axis integral containing a
single-axis ``k``-th derivative.

This construction is intentionally **axis-separable**. Mixed-derivative terms
are regarded as higher-order corrections and are not included in the main model.

---

## Residual backends

The module is split into three residual backends plus one dispatch layer.

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes`](@ref)

This backend handles Newton-Cotes residual extraction using exact rational
arithmetic.

Typical features include:

- exact composite coefficient handling,
- exact moment comparison,
- exact nonzero residual detection before final `Float64` conversion.

### [`Maranatha.ErrorEstimate.ErrorGauss`](@ref)

This backend handles Gauss-family residual extraction in `Float64`.

Residual moments are detected through tolerance-based comparisons rather than
exact arithmetic, reflecting the floating-point nature of the Gauss-family
construction.

### [`Maranatha.ErrorEstimate.ErrorBSpline`](@ref)

This backend applies the same floating-point residual philosophy to the
B-spline quadrature rules.

It supports both interpolation and smoothing spline families through midpoint-
shifted monomial probing on the composite grid.

### [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)

This layer provides the unified interface used by the rest of the package.

It is responsible for:

- normalizing backend outputs into a common residual representation,
- providing `1D` / `2D` / `3D` / `4D` estimators,
- providing generic `nd` estimators,
- exposing threaded variants where supported.

---

## Derivative backends

High-order derivative probes are evaluated through
[`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative`](@ref).

The derivative backend is selected explicitly via a method selector.
Currently supported backends include:

- `:forwarddiff`
- `:taylorseries`
- `:fastdifferentiation`
- `:enzyme`

The wrapper dispatches to the selected backend and reports an error if the
selector is invalid.

---

## Centering convention

Residuals are currently defined using a midpoint-centered convention.

The dispatch layer returns the center symbol explicitly so that downstream code
remains agnostic to the centering policy.

---

## Public API

The main public entry points are:

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)

Both interfaces:

- support leading-order only or LO+NLO+... collection through `nerr_terms`,
- dispatch to specialized low-dimensional implementations where available,
- fall back to generic multidimensional logic for higher dimensions,
- share the same residual-coefficient extraction pipeline.

---

## Design scope

This module is intentionally limited to truncation-error modeling.

It does **not**:

- perform least-``\chi^2`` fitting,
- define quadrature rules itself,
- provide rigorous interval-enclosure error bounds,
- hide tolerance choices inside unrelated layers.

That separation keeps the numerical responsibilities clearer across the
`Maranatha.jl` stack.

---

## Practical note

In the floating-point residual backends, the decision that a residual moment is
effectively nonzero depends on backend tolerance parameters. This affects
Gauss-family and B-spline rules, while Newton-Cotes residual detection uses
exact rational arithmetic.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate,
]
Private = true
```