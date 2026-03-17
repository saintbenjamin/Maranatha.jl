# Maranatha.ErrorEstimate.ErrorNewtonCotes

## Overview

`Maranatha.ErrorEstimate.ErrorNewtonCotes` provides the exact-rational midpoint
residual extraction backend for the Newton-Cotes family inside
`Maranatha.ErrorEstimate`.

Its job is to analyze the composite Newton-Cotes coefficient vector ``\beta``
and determine which midpoint-centered residual moments are truly nonzero.

Because the whole pipeline is based on `Rational{BigInt}`, this backend can make
that decision exactly rather than through floating-point tolerances.

---

## Core residual model

For a composite rule on the dimensionless grid ``u \in [0, N_{\texttt{sub}}]``,
the midpoint is

```math
c = \frac{N_{\texttt{sub}}}{2}.
```

For each order `k`, the backend compares:

- the exact shifted monomial moment
  ```math
  \int\limits_0^{N_{\texttt{sub}}} du \, (u-c)^k \, ,
  ```
- the quadrature-induced moment
  ```math
  \sum_{j=0}^{N_{\texttt{sub}}} \beta_j \, (j-c)^k.
  ```

Their difference defines the residual moment:

```math
\texttt{diff}_k
=
\int\limits_0^{N_{\texttt{sub}}} du \, (u-c)^k 
-
\sum_{j=0}^{N_{\texttt{sub}}} \beta_j (j-c)^k.
```

The corresponding Taylor-style coefficient is

```math
\texttt{coeff}_k = \frac{\texttt{diff}_k}{k!}.
```

This is the quantity used downstream in truncation-error modeling.

---

## Why exact arithmetic matters here

For Newton-Cotes rules, whether a residual is *exactly zero* is structurally
important: it determines the leading convergence order and which powers of `h`
appear in the modeled truncation error.

Using exact rational arithmetic means:

- no tolerance choice is needed,
- accidental near-zero floating-point artifacts are avoided,
- leading-order detection reflects the actual algebraic structure of the rule.

That makes this backend the cleanest residual detector among the currently
supported rule families.

---

## Function roles

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_midpoint_residual_term_from_beta`](@ref)

This is the lowest-level exact scanner. It assumes the composite coefficient
vector ``\beta`` is already available and returns only the first nonzero
midpoint residual term.

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_midpoint_residual_term`](@ref)

This wrapper starts from `(rule, boundary, Nsub)`, builds the exact composite
weights, and then delegates to the `β`-based scanner.

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_residual_ks_with_center`](@ref)

This helper collects only the residual orders `k`, together with the current
center tag `:mid`. It is useful when downstream logic needs the leading powers
but not the exact coefficients yet.

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_midpoint_residual_terms_from_beta`](@ref)

This is the multi-term exact scanner starting from an already assembled
coefficient vector ``\beta``.

### [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_midpoint_residual_terms`](@ref)

This is the higher-level convenience wrapper that starts from the user-facing
rule specification and returns multiple exact residual terms.

---

## Design note on centering

This backend currently uses only the midpoint-centered convention and therefore
returns `:mid` as the center tag.

The explicit center tag is still useful because it keeps the interface
compatible with future alternative centering policies.

---

## Scope note

This backend does not estimate derivatives, combine residuals with physical
midpoint probes, or assemble the final multidimensional truncation-error model.

Its responsibility is narrower:

- build or consume exact composite Newton-Cotes weights,
- test midpoint-shifted monomial residuals,
- return leading exact residual structure.

That separation keeps the exact residual logic isolated and easy to reason about.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.ErrorNewtonCotes,
]
Private = true
```