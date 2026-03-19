# ============================================================================
# src/ErrorEstimate/ErrorDispatch.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorDispatch

Unified error-estimation dispatch layer for `Maranatha.jl`.

`Maranatha.ErrorEstimate.ErrorDispatch` provides the primary public interface
for selecting and executing error estimators across all supported quadrature
families and estimation strategies.

It coordinates two major backend classes:

1. refinement-based estimators (derivative-free),
2. derivative-based estimators (direct or jet).

---

## Role in the overall workflow

Within the full `Maranatha` pipeline:

| Stage | Responsibility |
|:------|:---------------|
| Quadrature | compute numerical integral approximations |
| ErrorDispatch | select and execute error estimators |
| Fitter | perform extrapolation using estimates |
| Reporter | generate summaries and diagnostics |

This module acts as the gateway between the runner layer and the specialized
rule-family error estimators.

---

## Supported estimation strategies

### Refinement-based estimation

Selected when:

```julia
err_method == :refinement
```

Backend:

* [`ErrorDispatchRefinement.error_estimate_refinement`](@ref)

Key properties:

* derivative-free
* robust for non-smooth integrands
* based on coarse vs refined quadrature comparison
* supports Gauss, Newton–Cotes, and B-spline rules

---

### Derivative-based estimation

Derivative-based estimation is selected when
`err_method` is set to a derivative backend such as
`:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, or `:enzyme`.

Two sub-modes exist.

#### Direct derivative mode

Uses scalar derivative evaluations via the selected automatic-differentiation
backend:

* ForwardDiff
* TaylorSeries
* FastDifferentiation
* Enzyme

Backend:

* [`ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)

#### Jet-based mode

Activated by:

```julia
use_error_jet = true
```

Uses derivative jets to reuse high-order derivatives efficiently.

Backend:

* [`ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

Jet mode is typically faster when many derivative orders are required.

---

## Supported quadrature families

All supported families share the same unified interface:

* Gauss rules
* Newton–Cotes rules
* B-spline rules

Family-specific logic is delegated to specialized modules.

---

## Cache management

Derivative-based estimators internally use global caches:

* residual-model cache
* scalar derivative cache
* derivative-jet cache

These caches improve performance for repeated evaluations but are not
automatically cleared by this module itself.

Cache clearing is typically performed by higher-level orchestration
code (e.g., the runner) at the start of each dataset construction.

Cache management utilities are provided by the derivative dispatch layer.

---

## Typical usage

End users normally call:

```julia
error_estimate(...)
```

from the runner layer rather than invoking backend modules directly.

Example:

```julia
err = error_estimate(
    f, a, b, N, dim, rule, boundary;
    err_method = :forwarddiff,
    nerr_terms = 2,
    use_error_jet = false,
)
```

---

## Notes

* This module performs **strategy selection only**; it does not implement
  rule-specific error models itself.
* The returned object is the backend-specific named tuple describing the
  estimated error scale and associated metadata.
* For refinement-based estimation, derivative settings are ignored.
* For derivative-based estimation, the smoothing parameter `λ` is ignored.
"""
module ErrorDispatch

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..Quadrature.QuadratureUtils
import ..Quadrature.QuadratureDispatch
import ..ErrorEstimate.AutoDerivative.AutoDerivativeDirect
import ..ErrorEstimate.AutoDerivative.AutoDerivativeJet
import ..ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative
import ..ErrorEstimate.ErrorGauss.ErrorGaussDerivative
import ..ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative
import ..ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement
import ..ErrorEstimate.ErrorGauss.ErrorGaussRefinement
import ..ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement
import ..ErrorEstimate._RES_MODEL_CACHE
import ..ErrorEstimate._NTH_DERIV_CACHE
import ..ErrorEstimate._DERIV_JET_CACHE

include("ErrorDispatchDerivative.jl")
include("ErrorDispatchRefinement.jl")

using .ErrorDispatchDerivative
using .ErrorDispatchRefinement

"""
    error_estimate(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol = :refinement,
        nerr_terms::Int = 1,
        use_error_jet::Bool = false,
        λ = nothing,
        threaded_subgrid::Bool = false,
        real_type = nothing,
    )

Unified public dispatcher for all supported error-estimation backends.

# Function description
This function provides the main public entry point for the error-estimation
layer of `Maranatha.ErrorEstimate.ErrorDispatch`.

It selects one of the following backend families:

- refinement-based estimation via
  [`ErrorDispatchRefinement.error_estimate_refinement`](@ref)
- derivative-based direct estimation via
  [`ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
- derivative-based jet estimation via
  [`ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

The dispatch rule is:

- if `err_method == :refinement`, use the refinement backend
- otherwise, use the derivative backend selected by `use_error_jet`

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound.
- `b`:
  Upper integration bound.
- `N`:
  Number of subintervals per axis.
- `dim`:
  Number of dimensions.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `err_method::Symbol = :refinement`:
  Error-estimation backend selector.

  Supported meanings are:

  - `:refinement` for refinement-based estimation
  - derivative backend selectors such as
    `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, or `:enzyme`
- `nerr_terms::Int = 1`:
  Number of residual terms used by derivative-based estimators.
  Ignored when `err_method == :refinement`.
- `use_error_jet::Bool = false`:
  Selects the jet-based derivative estimator when `err_method != :refinement`.
- `λ = nothing`:
  Optional smoothing parameter passed through to the refinement backend.
  If `nothing`, zero is used in the active scalar type. This parameter is used
  only for smoothing B-spline refinement rules and is ignored by all other
  estimators.
- `threaded_subgrid::Bool = false`:
  Enables CPU threaded subgrid execution for refinement-based estimation.
  Ignored by derivative-based estimators.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and backend
  evaluation.

# Returns
- The named tuple returned by the selected backend.

# Errors
- Propagates backend-specific validation and computation errors.
- Throws if the chosen backend does not support the provided `rule`,
  `boundary`, or dimensionality.

# Notes
- This function is intended to be the single public dispatcher used by the
  runner layer.
- Derivative-cache clearing is intentionally not handled here; that remains the
  responsibility of higher-level orchestration code such as `run_Maranatha`.
"""
function error_estimate(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol = :refinement,
    nerr_terms::Int = 1,
    use_error_jet::Bool = false,
    λ = nothing,
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    if err_method === :refinement
        return ErrorDispatchRefinement.error_estimate_refinement(
            f,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            λ = λT,
            threaded_subgrid = threaded_subgrid,
            real_type = T,
        )
    end

    if use_error_jet
        return ErrorDispatchDerivative.error_estimate_derivative_jet(
            f,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    else
        return ErrorDispatchDerivative.error_estimate_derivative_direct(
            f,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    end
end

end  # module ErrorDispatch