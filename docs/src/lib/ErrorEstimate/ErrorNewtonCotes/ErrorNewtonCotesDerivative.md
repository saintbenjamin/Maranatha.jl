# Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative

## Overview

`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative` provides
the exact-rational midpoint-residual extraction backend for the
Newton–Cotes family inside `Maranatha.ErrorEstimate`.

Its role is to determine which midpoint-centered residual moments of a
composite Newton–Cotes rule are algebraically nonzero and to compute their
exact coefficients.

Unlike the Gauss and B-spline residual backends, this module operates entirely
in exact arithmetic using `Rational{BigInt}`. As a result, residual detection
does not rely on numerical tolerances and reflects the true algebraic structure
of the rule.

---

## Core residual model

For a composite rule on the dimensionless grid

```math
u \in [0, N_{\texttt{sub}}],
```

the midpoint is

```math
c = \frac{N_{\texttt{sub}}}{2}.
```

For each order `k`, the backend compares:

* the exact shifted monomial moment

  ```math
  \int_0^{N_{\texttt{sub}}} (u-c)^k \, du,
  ```
* the quadrature-induced moment produced by the composite rule.

Their difference defines the residual moment

```math
\texttt{diff}_k
=
\int_0^{N_{\texttt{sub}}} (u-c)^k \, du
-
\sum_j w_j (x_j-c)^k,
```

and the associated Taylor-style coefficient is

```math
\texttt{coeff}_k = \frac{\texttt{diff}_k}{k!}.
```

The backend returns the leading nonzero residual orders and coefficients
exactly.

---

## Why exact arithmetic matters

For Newton–Cotes rules, the presence or absence of a residual term is a
structural property that determines the formal convergence order of the rule.

Exact rational arithmetic ensures that:

* zero residuals are detected exactly,
* no tolerance selection is required,
* floating-point cancellation artifacts are avoided,
* leading-order detection matches the true polynomial exactness of the rule.

This makes the Newton–Cotes backend the most precise residual detector among
the supported quadrature families.

---

## Function roles

### Residual extraction helpers

Low-level routines construct the composite Newton–Cotes rule and evaluate
midpoint-shifted monomial residuals exactly. Both single-term and multi-term
scanners are provided.

### Order-only interface

Certain helpers expose only the detected residual orders together with a
center tag, allowing downstream code to determine leading powers without
carrying exact coefficients.

### Rule-based wrappers

Higher-level wrappers accept `(rule, boundary, Nsub)` and internally build the
composite rule before performing residual analysis.

---

## Design note on centering

This backend currently uses a midpoint-centered convention exclusively and
therefore reports the center tag `:mid`.

The explicit tag is retained for interface compatibility with other residual
backends and possible future centering policies.

---

## Scope note

This backend is responsible only for extracting exact residual structure for
Newton–Cotes rules.

It does not:

* evaluate derivatives,
* assemble multidimensional error models,
* perform refinement-based estimation,
* replace the Newton–Cotes quadrature construction layer.

Its output is intended for use by higher-level dispatch code that combines
residual data with physical derivative probes.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative,
]
Private = true
```