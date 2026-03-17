# ============================================================================
# src/ErrorEstimate/ErrorNewtonCotes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorNewtonCotes

Newton-Cotes–family error-estimation backend for `Maranatha.jl`.

`Maranatha.ErrorEstimate.ErrorNewtonCotes` provides error estimators
specialized for composite Newton-Cotes quadrature rules. It unifies two
complementary estimation strategies:

1. derivative-informed residual models based on exact midpoint moments,
2. refinement-based estimators based on coarse–versus–fine quadrature differences.

These estimators are tailored to composite Newton-Cotes rules with
boundary-dependent assembly constraints and are invoked automatically by
the higher-level error-dispatch layer.

---

## Role in the overall error-estimation architecture

Within the `Maranatha.ErrorEstimate` subsystem, this module implements the
Newton-Cotes branch of the estimator hierarchy:

| Layer | Responsibility |
|:------|:---------------|
| ErrorDispatch | selects rule-family backend |
| ErrorNewtonCotes | Newton-Cotes–specific estimators |
| ErrorNewtonCotesDerivative | derivative-based residual models |
| ErrorNewtonCotesRefinement | refinement-difference estimators |

---

## Available estimation strategies

### Derivative-based (residual) estimators

Provided by [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative`](@ref):

* Construct exact midpoint-centered residual expansions using rational arithmetic
* Use composite weights assembled from the Newton-Cotes rule
* Produce asymptotic truncation-error models involving derivatives of the integrand
* Especially suitable for smooth integrands and symbolic-quality analysis

Because Newton-Cotes rules use polynomial interpolation, the residual
structure can be expressed exactly in terms of rational coefficients,
enabling high-precision leading-term extraction.

---

### Refinement-based estimators

Provided by [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement`](@ref):

* Compare quadrature results at subdivision counts `N` and a boundary-compatible refined count
* Account for rule-specific validity constraints on composite subdivision sizes
* Require no derivative information
* Robust for non-smooth integrands or unstable derivative evaluation

Unlike Gauss rules, some Newton-Cotes boundary patterns restrict allowable
subdivision counts. The refinement backend automatically adjusts the refined
count to the nearest valid value when necessary.

---

## Typical usage

End users normally do not call this module directly.  
Instead, Newton-Cotes error estimation is selected automatically via:

```julia
Maranatha.ErrorEstimate.ErrorDispatch.error_estimate(...)
```

based on the chosen quadrature rule.

---

## Supported rules

All composite Newton-Cotes rules recognized by the quadrature layer are supported,
including rules of the form:

```julia
:newton_pK
```

where `K` denotes the degree of the interpolatory polynomial.

Boundary handling and valid-subdivision constraints are delegated to the
quadrature-dispatch layer.

---

## Notes

* All routines assume tensor-product integration over hypercubes `[a, b]^dim`.
* Returned values represent **effective error scales for downstream weighting**,
  not guaranteed strict truncation bounds.
* Derivative-based estimators use exact rational residual coefficients,
  while refinement-based estimators rely on empirical coarse–fine differences.
* Both strategies share a common interface, enabling uniform downstream
  fitting and convergence analysis workflows.
"""
module ErrorNewtonCotes

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.QuadratureDispatch

include("ErrorNewtonCotesDerivative.jl")
include("ErrorNewtonCotesRefinement.jl")

using .ErrorNewtonCotesDerivative
using .ErrorNewtonCotesRefinement

end  # module ErrorNewtonCotes