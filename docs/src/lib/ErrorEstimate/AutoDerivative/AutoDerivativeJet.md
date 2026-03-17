# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet

Jet-based derivative backend for truncation-error modeling in
`Maranatha.jl`.

---

## Overview

`AutoDerivativeJet` implements high-order derivative evaluation by propagating
a truncated series (a *jet*) through the integrand in a single pass.

Instead of computing derivatives one by one, this backend constructs a vector

```julia
[f(x), f'(x), f''(x), …, f^(n)(x)]
```

up to a requested maximum order.

This approach can dramatically reduce computational cost when many derivatives
are required.

---

## Method characteristics

Jet propagation evaluates the integrand once on a symbolic or dual-number
structure that encodes derivatives of all orders simultaneously.

Properties:

* amortizes function evaluations across derivative orders,
* typically much faster for high-order derivatives,
* produces a consistent set of derivatives at the same point,
* enables efficient reuse via caching.

Because derivatives are produced together, total cost grows more slowly than
linear in derivative order for many smooth functions.

---

## Backend selection

The jet backend is commonly selected when:

* many derivative orders are required,
* the integrand is smooth and compatible with jet propagation,
* performance is critical,
* repeated evaluations at the same point are expected.

---

## Numerical considerations

* Memory usage increases with requested order.
* Some functions or operations may not support jet propagation.
* Very high orders can amplify rounding error.
* Non-analytic behavior near the evaluation point may reduce accuracy.

Derivative jets are typically evaluated at the midpoint of the integration
interval.

---

## Relationship to direct evaluation

Compared with order-by-order computation:

* **Jet-based evaluation**

  * computes all derivatives simultaneously,
  * reduces repeated integrand work,
  * is often much faster for large orders,
  * may use more memory.

* **Direct evaluation**

  * computes derivatives individually,
  * works for a wider class of functions,
  * uses less memory,
  * may be slower when many orders are needed.

---

## Role in the error-estimation workflow

This backend supplies derivative jets to residual-based truncation-error
models. It does not perform quadrature, fitting, or refinement analysis.

Jet results may be cached and reused by the error-estimation dispatch layer to
avoid redundant computation.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet,
]
Private = true
```