# Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative

## Overview

`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative` provides the midpoint-residual extraction
backend for the Gauss-family quadrature rules inside `Maranatha.ErrorEstimate`.

Unlike the Newton-Cotes backend, this module works entirely in `Float64`.
Accordingly, the residual structure is detected numerically through tolerance
tests rather than exact rational arithmetic.

---

## Residual model

For a composite Gauss-family rule on the dimensionless grid
``u \in [0, N_{\text{sub}}]``, the midpoint is

```math
c = \frac{N_{\text{sub}}}{2}.
```

For each order `k`, the backend compares:

- the exact shifted monomial moment
  ```math
  M_k^{\texttt{exact}} = \int\limits_0^N du \, (u-c)^k \, ,
  ```
- the quadrature-induced moment
  ```math
  M_k^{\texttt{quad}} = \sum_i W_i \, (U_i-c)^k,
  ```
  where `(U, W)` comes from [`Maranatha.Quadrature.Gauss._composite_gauss_u_grid`](@ref).

Their difference defines the residual moment,

```math
\texttt{diff}_k = M_k^{\texttt{exact}} - M_k^{\texttt{quad}},
```

and the corresponding Taylor-style coefficient is

```math
\texttt{coeff}_k = \frac{\texttt{diff}_k}{k!}.
```

The backend returns the first few detected nonzero pairs ``(k, \texttt{coeff}_k)``.

---

## Why this backend is tolerance-based

For Gauss-family rules, the composite grid and weights are produced in floating
point. That means "exactly zero" residual detection is generally not available
in the same sense as in the rational Newton-Cotes backend.

Instead, this backend declares a residual nonzero only when

```math
|\texttt{diff}_k|
>
\texttt{tol\_abs} + \texttt{tol\_rel} \; |M_k^{\texttt{exact}}|.
```

This keeps the leading-order detection reasonably stable while avoiding obvious
false positives caused by roundoff.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative._exact_moment_shifted_float`](@ref)

This helper evaluates the exact shifted monomial moment in closed form, but in
`Float64`.

### [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative._leading_midpoint_residual_terms_gauss_float`](@ref)

This is the main backend entry point. It builds the composite Gauss grid,
compares exact and quadrature moments, applies the tolerance test, and returns
the first requested residual orders and coefficients.

---

## Boundary and family meaning

The `boundary` argument selects which Gauss family is used by the quadrature
backend:

- `:LU_EXEX` -> Gauss-Legendre
- `:LU_INEX` -> left Gauss-Radau
- `:LU_EXIN` -> right Gauss-Radau
- `:LU_ININ` -> Gauss-Lobatto

This backend does not derive those grids itself; it relies on the Gauss module
to produce the appropriate composite `u`-grid.

---

## Scope note

This backend is limited to residual-term extraction for Gauss-family rules.

It does not:

- compute physical derivative probes,
- assemble the final multidimensional truncation-error model,
- provide rigorous error bounds,
- replace the Gauss quadrature construction layer.

Its responsibility is simply to expose the leading midpoint residual structure
in a form that the higher-level dispatch code can reuse.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative,
]
Private = true
```