# ============================================================================
# src/ErrorEstimate/ErrorGauss.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorGauss

Gauss-family error-estimation backend for `Maranatha.jl`.

`Maranatha.ErrorEstimate.ErrorGauss` provides error estimators specialized
for tensor-product Gauss quadrature rules. It unifies two complementary
estimation strategies:

1. derivative-informed residual models based on midpoint moment analysis,
2. refinement-based estimators based on coarse–versus–fine quadrature differences.

These estimators are tailored to Gauss–Legendre–type composite rules and are
invoked automatically by the higher-level error-dispatch layer.

---

## Role in the overall error-estimation architecture

Within the `Maranatha.ErrorEstimate` subsystem, this module implements the
Gauss-family branch of the estimator hierarchy:

| Layer | Responsibility |
|:------|:---------------|
| ErrorDispatch | selects rule-family backend |
| ErrorGauss | Gauss-specific estimators |
| ErrorGaussDerivative | derivative-based residual models |
| ErrorGaussRefinement | refinement-difference estimators |

---

## Available estimation strategies

### Derivative-based (residual) estimators

Provided by [`ErrorGaussDerivative`](@ref):

* Analyze midpoint-centered residual moments of composite Gauss rules
* Construct asymptotic truncation-error models using derivatives
* Compatible with automatic-differentiation backends
  (`ForwardDiff`, `TaylorSeries`, `Enzyme`, etc.)
* Supports both direct derivative evaluation and jet-based reuse

These estimators are most effective when:

* the integrand is smooth,
* high-order derivatives exist and are stable,
* asymptotic convergence behavior is desired.

---

### Refinement-based estimators

Provided by [`ErrorGaussRefinement`](@ref):

* Compare quadrature results with subdivision counts `N` and `2N`
* Use the difference as an empirical error scale
* Require no derivative information
* Robust for non-smooth or difficult integrands

This approach is especially useful when:

* derivative evaluation is expensive or unreliable,
* the integrand contains localized structure,
* a conservative, data-driven error scale is sufficient.

---

## Typical usage

End users normally do not call this module directly.  
Instead, Gauss-family error estimation is selected automatically via:

```julia
Maranatha.ErrorEstimate.ErrorDispatch.error_estimate(...)
```

based on the chosen quadrature rule.

---

## Supported rules

All composite Gauss rules recognized by the quadrature layer are supported,
including rules of the form:

```julia
:gauss_pK
```

where `K` denotes the number of nodes per subinterval.

Boundary handling is delegated to the quadrature-dispatch layer.

---

## Notes

* All routines assume tensor-product integration over hypercubes
  `[a, b]^dim`.
* Returned values represent **effective error scales for downstream weighting**,
  not guaranteed strict truncation bounds.
* The derivative-based and refinement-based strategies share a common interface,
  enabling uniform use in fitting and convergence analysis workflows.
"""
module ErrorGauss

import ..JobLoggerTools
import ..Quadrature.Gauss
import ..Quadrature.QuadratureDispatch

include("ErrorGaussDerivative.jl")
include("ErrorGaussRefinement.jl")

using .ErrorGaussDerivative
using .ErrorGaussRefinement

end  # module ErrorGauss