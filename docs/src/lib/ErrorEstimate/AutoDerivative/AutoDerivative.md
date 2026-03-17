# Maranatha.ErrorEstimate.AutoDerivative

Automatic-derivative backend layer used by the residual-based error estimators
in `Maranatha.jl`.

---

## Overview

`Maranatha.ErrorEstimate.AutoDerivative` provides a unified interface for
evaluating high-order derivatives required by truncation-error models.
It encapsulates multiple differentiation strategies behind a consistent API,
allowing the error-estimation layer to request derivatives without committing
to a specific backend.

The module is designed for:

- high-order midpoint derivative probes,
- stable derivative evaluation across a wide class of integrands,
- reuse via caching,
- backend switching without altering higher-level logic.

It does not perform quadrature or fitting.

---

## Supported derivative strategies

Two complementary approaches are implemented.

### Direct derivative evaluation

Computes each derivative order independently using automatic differentiation
or finite-difference fallback methods.

Characteristics:

- simple control flow,
- low memory footprint,
- suitable for small derivative orders,
- robust for functions without efficient Taylor propagation.

### Jet-based derivative evaluation

Computes a full vector of derivatives in a single pass using Taylor-series
propagation or equivalent jet techniques.

Characteristics:

- efficient for high derivative orders,
- avoids repeated reevaluation of the integrand,
- naturally produces derivative sequences
  `[f(x), f'(x), f''(x), …]`,
- may impose additional constraints on the integrand.

---

## Backend policy

Backend selection is controlled by the calling error-estimation layer.
Typical policies include:

- prefer jet evaluation when many derivatives are required,
- fall back to direct evaluation when jet construction fails or is unsupported,
- ensure finite results before use in residual models.

Derivative values are normally evaluated at the physical midpoint of the
integration domain.

---

## Caching behavior

To reduce computational cost, previously computed derivatives or derivative
jets may be stored in global caches maintained by the parent
`ErrorEstimate` module.

Caching is keyed by function identity, evaluation point, derivative order,
and backend tag.

---

## Numerical considerations

- Very high derivative orders can be expensive or numerically unstable.
- Non-smooth integrands may degrade derivative accuracy.
- Floating-point overflow or loss of significance may occur for extreme orders.
- Jet-based methods can amplify rounding effects if the integrand has large curvature.

These tools are intended for asymptotic error modeling rather than rigorous
bounds.

---

## Relationship to refinement-based estimation

Refinement-based error estimators do not use derivatives at all.
This module is only required for residual-based truncation-error modeling.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative,
]
Private = true
```