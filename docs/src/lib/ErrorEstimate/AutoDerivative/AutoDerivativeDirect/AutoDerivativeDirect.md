# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect

Direct derivative-evaluation backend for truncation-error modeling in
`Maranatha.jl`.

---

## Overview

`AutoDerivativeDirect` implements order-by-order derivative evaluation.
Each requested derivative is computed independently rather than as part of a
full derivative jet.

This backend prioritizes robustness and general applicability over maximal
throughput.

It is particularly useful when:

- only a small number of derivatives is needed,
- jet construction is expensive or unsupported,
- the integrand is not compatible with Taylor propagation,
- predictable memory usage is required.

---

## Method characteristics

Direct evaluation typically proceeds by applying an automatic-differentiation
backend (or fallback strategy) separately for each derivative order.

Properties:

- no requirement to propagate high-order series objects,
- minimal intermediate storage,
- stable for functions with complex control flow,
- straightforward error handling per derivative.

Because each derivative is computed independently, computational cost grows
approximately linearly with derivative order.

---

## Backend selection

The direct backend is commonly selected when:

- jet-based evaluation is disabled or unavailable,
- derivative orders are modest,
- the integrand includes operations unsupported by jet methods,
- stability is preferred over raw performance.

---

## Numerical considerations

- Repeated evaluations can accumulate rounding error.
- High derivative orders may still become expensive.
- Non-smooth behavior near the evaluation point can degrade accuracy.
- Overflow or non-finite results may occur for extreme orders.

Derivative probes are typically evaluated at the midpoint of the integration
interval.

---

## Relationship to jet-based evaluation

Compared with jet-based methods:

- **Direct evaluation**
  - computes derivatives one at a time,
  - uses less memory,
  - works for a broader class of functions,
  - may be slower for large derivative orders.

- **Jet-based evaluation**
  - computes many derivatives simultaneously,
  - amortizes integrand evaluations,
  - can be significantly faster when many orders are required.

---

## Role in the error-estimation workflow

This backend feeds derivative values into residual-based truncation-error
models. It does not construct the models itself and does not perform
quadrature or refinement comparisons.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect,
]
Private = true
```