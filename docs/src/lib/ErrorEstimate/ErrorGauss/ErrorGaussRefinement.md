# Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement

## Overview

`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement` provides the
refinement-based error-estimation backend for Gauss-family quadrature rules
inside `Maranatha.ErrorEstimate`.

Unlike the residual-based backend
[`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative`](@ref), which analyzes
midpoint residual moments of a fixed rule, this module estimates truncation
error by directly comparing quadrature evaluations at two resolutions.

The refinement approach is purely numerical and does not require:

- residual-moment analysis,
- derivative evaluation,
- symbolic or automatic differentiation,
- knowledge of asymptotic error coefficients.

It is therefore particularly suitable when derivative-based modeling is
expensive, unstable, or conceptually mismatched to the quadrature family.

---

## Refinement model

Let

```math
Q_N
```

denote the quadrature estimate obtained with subdivision count `N`, and let

```math
Q_{2N}
```

denote the estimate computed using a refined grid with twice as many
subdivisions along each axis.

The refinement backend forms the difference

```math
\Delta_N = Q_{2N} - Q_N,
```

and uses

```math
|\Delta_N|
```

as a practical error-scale estimate.

Under smooth convergence,

```math
Q_N = I_{\mathrm{exact}} + C h^p + O(h^{p+1}),
```

so that

```math
\Delta_N \approx C h^p (2^{-p} - 1).
```

The backend does not attempt to infer `p` or normalize the estimate; it reports
the raw refinement difference.

---

## Multidimensional refinement

For tensor-product Gauss rules in dimension `d`, refinement doubles the number
of subdivisions per axis:

```math
N \rightarrow 2N .
```

The number of evaluation points therefore grows by roughly `2^d`, which can
become expensive in high dimension. However, the method remains attractive
because it:

* requires no derivative probes,
* uses the same quadrature machinery,
* behaves robustly for smooth integrands.

---

## Implementation behavior

The backend:

1. validates inputs for Gauss-family rules,
2. evaluates the quadrature at subdivision count `N`,
3. evaluates the quadrature at subdivision count `2N`,
4. returns a structured result containing both values and the refinement
   difference.

No Richardson-style extrapolation or asymptotic coefficient extraction is
performed.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement._estimate_by_refinement_gauss`](@ref)

Internal helper implementing the coarse-versus-refined comparison and producing
a detailed result record, including both quadrature values and mesh sizes.

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement.error_estimate_refinement_gauss`](@ref)

Public entry point used by the higher-level dispatch layer. This function
selects the appropriate dimensional specialization and forwards the request to
the internal refinement estimator.

---

## Boundary and family meaning

The `boundary` argument selects which Gauss-family rule is used:

* `:LU_EXEX` → Gauss–Legendre
* `:LU_INEX` → left Gauss–Radau
* `:LU_EXIN` → right Gauss–Radau
* `:LU_ININ` → Gauss–Lobatto

Quadrature grids are constructed by the Gauss module; this backend only
coordinates the refinement comparison.

---

## Numerical characteristics

Compared with residual-based estimators, the refinement approach:

**Advantages**

* derivative-free,
* robust for complex integrands,
* simple conceptual model,
* insensitive to residual-detection tolerances.

**Limitations**

* requires additional quadrature evaluations,
* computational cost grows as `2^d`,
* does not provide asymptotic error coefficients,
* produces an empirical error scale rather than a bound.

---

## Scope note

This backend provides refinement-based error indicators only.

It does not:

* analyze residual moments,
* construct asymptotic truncation-error models,
* compute derivative-based estimates,
* replace the Gauss quadrature construction layer.

Its role is to supply a practical error scale derived from resolution refinement.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement,
]
Private = true
```