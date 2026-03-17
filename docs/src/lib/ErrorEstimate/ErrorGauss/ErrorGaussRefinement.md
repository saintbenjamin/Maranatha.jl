# Maranatha.ErrorEstimate.ErrorGaussRefine

## Overview

`Maranatha.ErrorEstimate.ErrorGaussRefine` provides the refinement-based
error-estimation backend for Gauss-family quadrature rules inside
`Maranatha.ErrorEstimate`.

Unlike the residual-based backend
[`Maranatha.ErrorEstimate.ErrorGauss`](@ref), which analyzes midpoint residual
moments of a fixed rule, this module estimates truncation error by comparing
results at multiple resolutions.

The refinement approach is purely numerical and does not require:

- residual-moment analysis,
- derivative evaluation,
- symbolic or automatic differentiation,
- knowledge of the leading asymptotic error coefficient.

It is therefore particularly suitable when:

- the integrand is expensive to differentiate,
- high-order derivatives are unreliable,
- residual modeling assumptions are difficult to validate,
- a fast empirical error estimate is desired.

---

## Refinement model

Let

```math
I_h
```

denote the quadrature estimate obtained with subinterval size `h`, and let

```math
I_{h/2}
```

denote the estimate computed using a refined grid with twice the resolution
(along each axis).

The refinement backend uses the difference

```math
\Delta_h = I_{h/2} - I_h
```

as a proxy for the truncation error at scale `h`.

Under standard assumptions for structured composite rules,

```math
I_h = I_{\text{exact}} + C h^p + O(h^{p+1}),
```

so that

```math
\Delta_h \approx C h^p (2^{-p} - 1).
```

Although the exact asymptotic coefficient is unknown, the magnitude of
`Δ_h` provides a practical estimate of the discretization error.

This strategy avoids explicit modeling of the residual structure and instead
relies on empirical convergence behavior.

---

## Multidimensional refinement

For tensor-product rules in dimension `d`, refinement doubles the number of
subintervals along each axis.

If the original resolution uses `N` subdivisions per axis, the refined grid uses

```math
2N \quad \text{per axis}.
```

Consequently, the total number of evaluation points increases by approximately

```math
2^d,
```

which can become expensive in higher dimensions. Nevertheless, the approach
remains attractive because it:

* requires no derivative probes,
* uses the same quadrature construction machinery,
* behaves robustly for a wide class of smooth integrands.

---

## Relation to Richardson-style ideas

The refinement estimator is conceptually related to Richardson extrapolation,
but it stops short of constructing an extrapolated limit.

Instead of estimating the exact integral, it only estimates the scale of the
leading truncation error by measuring how the quadrature result changes under
grid refinement.

No attempt is made to infer the convergence order `p` automatically in this
backend.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorGaussRefine.error_estimate_gauss`](@ref)

This is the main backend entry point for Gauss-family rules.

It:

1. constructs the quadrature estimate at resolution `N`,
2. constructs the estimate at resolution `2N`,
3. computes their difference,
4. packages the result as an error estimate object compatible with the
   higher-level dispatch layer.

The actual quadrature construction is delegated to the Gauss quadrature module;
this backend only orchestrates the comparison.

---

## Boundary and family meaning

The `boundary` argument selects which Gauss-family rule is used:

* `:LU_EXEX` -> Gauss-Legendre
* `:LU_INEX` -> left Gauss-Radau
* `:LU_EXIN` -> right Gauss-Radau
* `:LU_ININ` -> Gauss-Lobatto

As in the residual backend, this module does not construct the quadrature grids
directly; it relies on the Gauss quadrature layer to supply the composite rule.

---

## Numerical characteristics

Compared with residual-based estimators, the refinement approach:

**Advantages**

* derivative-free,
* robust for complicated integrands,
* simple conceptual model,
* often very fast in practice when derivative evaluation is costly,
* insensitive to residual-detection tolerances.

**Limitations**

* requires additional quadrature evaluations,
* computational cost grows as `2^d` in dimension `d`,
* does not provide asymptotic error coefficients,
* may underestimate error when convergence is irregular.

---

## Scope note

This backend is limited to refinement-based error estimation for
Gauss-family rules.

It does not:

* analyze residual moments,
* construct truncation-error series,
* compute derivative-based models,
* replace the Gauss quadrature construction layer.

Its responsibility is solely to provide an empirical error estimate based on
resolution refinement.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorGaussRefine,
]
Private = true
```