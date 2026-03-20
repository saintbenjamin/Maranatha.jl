# Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement

## Overview

`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement` provides the refinement-based
error-estimation backend for the Newton-Cotes family inside
`Maranatha.ErrorEstimate`.

In contrast to
[`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative`](@ref), which analyzes exact
midpoint residual moments of the composite coefficient vector ``\beta``, this
module estimates truncation error empirically by comparing results obtained at
different resolutions.

The refinement approach does not depend on:

- exact rational residual analysis,
- symbolic structure of the rule,
- derivative evaluation,
- asymptotic error coefficients.

Instead, it measures how the quadrature result changes under grid refinement.

---

## Refinement model

Let

```math
I_h
```

denote the composite Newton-Cotes estimate with subinterval size `h`, and let

```math
I_{h/2}
```

be the estimate obtained using twice the resolution.

The refinement backend uses

```math
\Delta_h = I_{h/2} - I_h
```

as a practical estimate of the truncation error at scale `h`.

For sufficiently smooth integrands,

```math
I_h = I_{\text{exact}} + C h^p + O(h^{p+1}),
```

which implies

```math
\Delta_h \approx C h^p (2^{-p} - 1).
```

Although the constant `C` and order `p` are not explicitly determined, the
magnitude of `Δ_h` provides a useful empirical measure of discretization error.

---

## Why refinement is useful for Newton-Cotes rules

The exact residual backend determines the formal algebraic error structure of a
rule, but it does not account for:

* floating-point roundoff,
* integrand irregularities,
* cancellation effects,
* practical convergence behavior on finite grids.

The refinement approach complements the exact method by providing a purely
numerical estimate that reflects the actual computed result.

This is particularly valuable when:

* the integrand is not sufficiently smooth,
* high-order asymptotics are not yet dominant,
* the rule order is high and coefficients become large,
* numerical stability is a concern.

---

## Multidimensional refinement

For tensor-product Newton-Cotes rules in dimension `d`, refinement doubles the
number of subintervals along each axis.

If the original grid uses `N` subdivisions per axis, the refined grid uses

```math
2N.
```

The total number of evaluation points therefore increases by approximately

```math
2^d,
```

which can become expensive in high dimensions but remains straightforward to
implement because it reuses the same quadrature construction logic.

---

## Relation to Richardson-style ideas

The refinement estimator is related to Richardson extrapolation but does not
attempt to estimate the exact integral.

It measures only the change between successive resolutions rather than fitting
a convergence model.

No attempt is made to infer the formal convergence order of the Newton-Cotes
rule in this backend.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement.error_estimate_refinement_newton_cotes`](@ref)

This is the primary backend entry point.

It:

1. obtains the coarse composite Newton-Cotes estimate at resolution `N`,
   either by computing it internally or by reusing a caller-supplied
   `I_coarse`,
2. computes the refined estimate at a boundary-compatible refined subdivision
   count,
3. forms the difference between the two,
4. packages the result into an error-estimate object compatible with the
   higher-level dispatch layer.

The actual construction of the quadrature weights is delegated to the
Newton-Cotes quadrature module.

---

## Boundary interpretation

The `boundary` argument selects the Newton-Cotes variant used by the quadrature
backend (e.g., open or closed endpoint treatment).

This backend does not analyze the boundary structure directly; it simply uses
the rule specified by the quadrature layer.

---

## Numerical characteristics

Compared with the exact residual backend, the refinement approach:

**Advantages**

* derivative-free,
* insensitive to algebraic residual structure,
* robust to floating-point effects,
* reflects actual convergence behavior,
* simple conceptual model.

**Limitations**

* requires additional quadrature evaluations,
* computational cost grows exponentially with dimension,
* does not expose the formal error order,
* may underestimate error when convergence is irregular.

When an already computed coarse quadrature value is available, the backend can
reuse it through `I_coarse` to avoid redundant coarse-grid work.

---

## Design contrast with the exact backend

`ErrorNewtonCotesDerivative` answers:

> *What is the formal algebraic truncation structure of the rule?*

`ErrorNewtonCotesRefinement` answers:

> *How much does the computed result change when the grid is refined?*

Both perspectives are useful, but they serve different purposes.

---

## Scope note

This backend is limited to refinement-based error estimation for
Newton-Cotes rules.

It does not:

* analyze midpoint residual moments,
* construct exact rational coefficients,
* evaluate derivatives,
* assemble multidimensional truncation-error models.

Its role is purely empirical: estimate discretization error by resolution
comparison.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement,
]
Private = true
```