# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module AutoDerivative

Unified automatic-differentiation support layer for the
`Maranatha.jl` error-estimation subsystem.

`Maranatha.AutoDerivative` provides backend-agnostic routines for computing
scalar derivatives and derivative jets required by the derivative-based
error models. It combines two complementary differentiation strategies:

- **Direct derivatives** via repeated differentiation
- **Derivative jets** for efficient multi-order evaluation at a single point

These implementations are organized into the submodules

- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect`](@ref): scalar `n`-th derivative computation
- [`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet`](@ref): derivative-jet construction

Both submodules share global caches defined in
`Maranatha.ErrorEstimate`, enabling reuse of expensive derivative results
across the quadrature pipeline.

---

## Role in the overall workflow

This module is not a user-facing API. It acts as an internal infrastructure
layer between

- the error-estimation dispatchers in `Maranatha.ErrorEstimate`, and
- external differentiation backends such as
  `ForwardDiff`, `TaylorSeries`, and `Enzyme`.

A typical call chain is

```

run_Maranatha
→ ErrorEstimate.ErrorDispatch
→ derivative-based estimator
→ AutoDerivative.nth_derivative / derivative_jet
→ backend-specific AD library

```

---

## Supported differentiation backends

Both direct and jet pathways support multiple differentiation engines:

- `:forwarddiff`         → Forward-mode automatic differentiation
- `:taylorseries`        → Truncated Taylor expansion
- `:enzyme`              → Reverse-mode AD via Enzyme

Backend selection is controlled externally through the `err_method`
parameter passed to the error-estimation layer.

---

## Direct vs. jet-based derivatives

Two complementary interfaces are provided:

### Direct derivatives

Compute a single derivative of order `n`:

```math
f^{(n)}(x)
```

This is appropriate when only a few derivative orders are required.

---

### Derivative jets

Compute all derivatives up to order `nmax` simultaneously:

```math
[f(x), f'(x), f''(x), \\ldots, f^{(nmax)}(x)]
```

Jets are typically more efficient when several derivative orders at the same
point are needed, since the underlying computation can be shared.

---

## Caching behavior

All derivative results are cached in shared global dictionaries:

* [`_NTH_DERIV_CACHE`](@ref): scalar derivative cache
* [`_DERIV_JET_CACHE`](@ref): jet cache
* [`_RES_MODEL_CACHE`](@ref): higher-level residual-model cache

The caches are keyed by callable identity, evaluation point, derivative order,
and backend symbol, allowing reuse across resolutions and dimensions.

Cache management (including clearing policies) is handled by the
error-estimation layer rather than this module.

---

## Notes

* This module assumes scalar real-to-real callables.
* It does not perform quadrature or error estimation itself.
* Numerical stability checks and fallback strategies are implemented at higher
  layers of the package.
* Backend-specific limitations (e.g., symbolic compatibility requirements) are
  propagated to the caller.

---

## See also

* [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)
* [`Maranatha.Runner.run_Maranatha`](@ref)
* External packages: ForwardDiff.jl, TaylorSeries.jl,
  Enzyme.jl
"""
module AutoDerivative

import ..JobLoggerTools
import ..ErrorEstimate._RES_MODEL_CACHE
import ..ErrorEstimate._NTH_DERIV_CACHE
import ..ErrorEstimate._DERIV_JET_CACHE

include("AutoDerivativeDirect/AutoDerivativeDirect.jl")
include("AutoDerivativeJet/AutoDerivativeJet.jl")

using .AutoDerivativeDirect
using .AutoDerivativeJet

end  # module AutoDerivative
