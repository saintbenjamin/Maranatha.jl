# Maranatha.ErrorEstimate.ErrorGauss

Gauss-family truncation-error modeling backends for `Maranatha.jl`.

---

## Overview

`ErrorGauss` implements truncation-error estimators specialized for
Gauss-type quadrature rules within the `Maranatha.jl` framework.

Gauss rules differ structurally from Newton–Cotes rules: nodes are not
equally spaced, weights are nonuniform, and exactness properties depend on
orthogonal-polynomial constructions rather than algebraic interpolation.
Consequently, dedicated residual extraction and refinement logic are required.

This module provides those rule-family-specific components.

---

## Supported estimation frameworks

Two complementary approaches are implemented.

### Residual-based (derivative) estimation

Constructs an asymptotic truncation-error model by combining:

- rule-specific residual structure,
- leading nonzero moment orders,
- midpoint derivative probes of the integrand.

Because Gauss rules exhibit high algebraic exactness, the leading residual
orders are typically high, and the resulting error models can decay rapidly
with resolution.

Residual extraction for Gauss rules is performed numerically in `Float64`.

---

### Refinement-based estimation

Uses direct comparison between coarse and refined quadrature evaluations.

Typical procedure:

1. Evaluate the Gauss rule with subdivision count `N`.
2. Re-evaluate with a refined subdivision (e.g., `2N`).
3. Use the difference as an empirical truncation-error scale.

This approach is derivative-free and robust for difficult integrands.

---

## Architecture

The module is split into two internal components:

- **Derivative backend**  
  Implements residual extraction and derivative-based error models for
  Gauss rules.

- **Refinement backend**  
  Implements coarse-versus-refined difference estimators.

Both are accessed through the unified dispatch layer
[`Maranatha.ErrorEstimate.ErrorDispatch`](@ref).

---

## Numerical characteristics

- Residual coefficients are computed numerically rather than exactly.
- Leading error orders can be high due to the exactness of Gauss rules.
- Performance is generally excellent for smooth integrands.
- Stability may degrade for highly oscillatory or non-smooth functions.
- Computational cost increases with dimension in tensor-product settings.

---

## Role in the package

`ErrorGauss` supplies Gauss-specific components to the global
error-estimation pipeline.

Users typically do not call this module directly; instead, it is invoked
automatically when a Gauss-family quadrature rule is selected.

---

## Notes

- Estimates represent error scales, not strict bounds.
- Residual-based and refinement-based results may differ in interpretation.
- Behavior depends strongly on integrand smoothness and dimensionality.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorGauss,
]
Private = true
```