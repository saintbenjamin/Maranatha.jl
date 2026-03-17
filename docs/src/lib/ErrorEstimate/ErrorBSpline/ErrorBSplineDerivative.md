# Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative

## Overview

`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative` provides the midpoint-residual extraction
backend for B-spline quadrature rules inside `Maranatha.ErrorEstimate`.

Like the Gauss residual backend, it works entirely in `Float64` and therefore
detects nonzero residual structure through tolerance tests rather than exact
arithmetic.

---

## Residual model

For a composite B-spline quadrature rule on the dimensionless interval
``u \in [0, N_{\texttt{sub}}]``, the midpoint is

```math
c = \frac{N_{\texttt{sub}}}{2}.
```

For each scanned order ``k``, the backend compares:

- the exact shifted monomial moment
  ```math
  \int\limits_0^N du \, (u-c)^k \, ,
  ```
- the quadrature-induced moment
  ```math
  \sum_i w_i \, (x_i-c)^k \, .
  ```

Their difference defines the residual moment,

```math
\texttt{diff}_k
=
\int\limits_0^N du \, (u-c)^k
-
\sum_i w_i \, (x_i-c)^k,
```

and the associated Taylor-style coefficient is

```math
\texttt{coeff}_k = \frac{\texttt{diff}_k}{k!}.
```

The backend returns the first few detected nonzero pairs ``(k, \texttt{coeff}_k)``.

---

## Supported spline families

This backend supports both spline rule families defined by the quadrature layer:

- interpolation splines: `:bspline_interp_p2`, `:bspline_interp_p3`, ...
- smoothing splines: `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

For smoothing splines, the residual detector forwards the smoothing strength
``\lambda`` to the spline quadrature constructor.

---

## Why detection is tolerance-based

Because the spline nodes and weights are generated in floating point, residual
moments cannot generally be tested for exact zero in an algebraic sense.

Instead, this backend treats a residual as nonzero only when it exceeds a mixed
absolute/relative threshold. This avoids classifying roundoff artifacts as
genuine leading residual structure.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative._exact_moment_shifted_float`](@ref)

This helper evaluates the exact shifted monomial moment in closed form, but in
`Float64`.

### [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative._leading_midpoint_residual_terms_bspline_float`](@ref)

This is the main residual extractor. It constructs the spline quadrature on the
dimensionless interval, compares exact and quadrature moments, applies the
tolerance test, and returns the first requested residual orders and
coefficients.

### [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative._leading_residual_ks_with_center_bspline_float`](@ref)

This wrapper keeps only the detected residual orders and returns the center tag
`:mid`, matching the interface expected by the higher-level dispatch layer.

---

## Design note

This backend constructs the spline quadrature directly on ``[0, N_{\texttt{sub}}]``
so that the dimensionless interval used in residual probing matches the tiling
parameter used elsewhere in the error-model pipeline.

That keeps the midpoint-centered moment tests aligned with the intended
composite-grid interpretation.

---

## Scope note

This backend is responsible only for exposing leading midpoint residual
structure for B-spline quadrature rules.

It does not:

- compute physical derivative probes,
- assemble the final multidimensional error model,
- provide rigorous error bounds,
- replace the spline quadrature construction itself.

Its output is meant to feed the higher-level residual and error-estimation
dispatch code.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative,
]
Private = true
```