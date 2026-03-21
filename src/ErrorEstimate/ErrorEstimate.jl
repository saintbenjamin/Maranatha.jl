# ============================================================================
# src/ErrorEstimate/ErrorEstimate.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorEstimate

Residual-based and refinement-based truncation-error modeling engine used by
`Maranatha.jl`.

# Overview

`Maranatha.ErrorEstimate` provides the truncation-error modeling layer for the
structured tensor-product quadrature backends in `Maranatha.jl`.

It implements two complementary frameworks:

- a residual-based asymptotic truncation-error model, which extracts leading
  nonzero midpoint-centered residual moments of composite quadrature rules and
  combines them with high-order derivative probes at the physical midpoint, and
- a refinement-based error model, which compares coarse and refined quadrature
  evaluations directly without requiring high-order derivative probes.

These tools are intended for:

- fit stabilization,
- error-scale diagnostics,
- automatic leading-order detection from residual structure,
- fast quadrature-difference based error estimation for selected rule families.

They are not intended to provide strict error bounds.

------------------------------------------------------------------------

# Core idea

## Residual-based branch

1. Build a composite quadrature rule on a dimensionless tiling grid.
2. Detect leading nonzero residual moment orders from the rule structure.
3. Convert those residuals into factorial-scaled coefficients.
4. Combine them with midpoint derivative probes to construct a truncation-error model.

The residual-based model is axis-separable by construction; mixed-derivative
contributions are treated as higher-order effects and are intentionally omitted.

## Refinement-based branch

1. Evaluate the quadrature rule on a coarse subdivision count.
2. Re-evaluate the same rule on a refined subdivision count.
3. Use the coarse-versus-refined difference as a practical error-scale estimate.

This branch is intended as a lightweight alternative when derivative-based
probing is expensive, unstable, or conceptually mismatched to the quadrature
construction.

------------------------------------------------------------------------

# Architecture

The module is structured into residual backends, refinement backends, and
unified dispatch layers.

## Residual backends

### 1. [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesDerivative`](@ref)

Exact-rational residual extraction for Newton-Cotes rules.

### 2. [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussDerivative`](@ref)

Configurable-real-type residual extraction for Gauss-family rules.

### 3. [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineDerivative`](@ref)

Configurable-real-type residual extraction for B-spline quadrature rules.

### 4. [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative`](@ref)

Unified dispatch layer for residual extraction, derivative probing, and
dimension-specific / dimension-generic residual-based error estimators.

## Refinement backends

### 5. [`Maranatha.ErrorEstimate.ErrorNewtonCotes.ErrorNewtonCotesRefinement`](@ref)

Refinement-difference error estimation for Newton-Cotes rules.

### 6. [`Maranatha.ErrorEstimate.ErrorGauss.ErrorGaussRefinement`](@ref)

Refinement-difference error estimation for Gauss-family rules.

### 7. [`Maranatha.ErrorEstimate.ErrorBSpline.ErrorBSplineRefinement`](@ref)

Refinement-difference error estimation for B-spline quadrature rules.

### 8. [`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchRefinement`](@ref)

Unified dispatch layer for rule-family refinement-based error estimation.

------------------------------------------------------------------------

# Derivative policy

For the residual-based branch, high-order derivatives are obtained through the
shared automatic-differentiation layer provided by
[`Maranatha.ErrorEstimate.AutoDerivative`](@ref).

Two complementary derivative strategies are supported:

- a direct evaluation backend, which computes derivatives individually on demand, and
- a jet-based backend, which computes a vector of derivatives in a single pass.

These are exposed through:

- [`ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
- [`ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

Both strategies evaluate derivatives at the physical midpoint and share
common caching and backend-safety policies.

The refinement-based branch does not use derivative probes; instead, it
constructs an error estimate directly from the difference between coarse and
refined quadrature evaluations.

------------------------------------------------------------------------

# Public API

Primary entry points include:

- [`ErrorDispatch.error_estimate`](@ref)
- [`ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref)
- [`ErrorDispatch.ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)
- [`ErrorDispatch.ErrorDispatchRefinement.error_estimate_refinement`](@ref)

These interfaces support dimension-specific and generic multidimensional
dispatch, depending on the selected backend.

Shared scalar `rule` / `boundary` symbols and axis-wise tuple or vector
specifications are both supported throughout the quadrature-facing public
dispatchers. For refinement-based estimation, axis-wise `rule` specifications
must belong to a single quadrature family on all axes.

------------------------------------------------------------------------

# Notes

- This module models truncation error for structured composite rules; it does not perform fitting itself.
- Residual-based estimators and refinement-based estimators are complementary and may differ in cost, stability, and asymptotic interpretation.
- Numerical decisions in `Float64` residual detection are controlled by backend tolerances.
- Computational cost grows rapidly with dimension in generic multidimensional estimators.
- Refinement-based estimators can be especially useful when high-order derivative evaluation is too expensive or theoretically undesirable for a given quadrature family.
"""
module ErrorEstimate

import ..LinearAlgebra

import ..Utils.JobLoggerTools
import ..Utils.QuadratureBoundarySpec
import ..Quadrature

"""
    _RES_MODEL_CACHE::Dict{Tuple, Tuple}

Global cache for residual-model data keyed by
`(rule, boundary, nterms, kmax, real_type)`.

# Description
This cache stores the tuple returned by
[`ErrorDispatch.ErrorDispatchDerivative._get_residual_model_fixed`](@ref), namely:

- `ks`: leading residual indices,
- `coeffs`: corresponding residual coefficients,
- `center`: centering convention tag.

The purpose of this cache is to avoid recomputing the same residual model
for repeated error-estimation calls using identical quadrature settings.

# Notes
- The cache key intentionally excludes `Nref`, because the fixed residual model
  is currently treated as depending only on
  `(rule, boundary, nterms, kmax, real_type)`.
- Cached values are stored in the exact form returned by
  [`ErrorDispatch.ErrorDispatchDerivative._leading_residual_terms_any`](@ref).
"""
const _RES_MODEL_CACHE = Dict{Tuple, Tuple}()

"""
    _NTH_DERIV_CACHE::Dict{Tuple{UInt,Float64,Int,Symbol},Float64}

Global cache for scalar derivative evaluations.

# Description
This cache stores previously computed `n`th-derivative values so that repeated
calls with the same function identity, evaluation point, derivative order,
backend symbol, and active scalar type can reuse an earlier result.

# Key structure
Each key has the form:

- `UInt`: encoded function identity via `objectid`,
- evaluation point in the active scalar type,
- `Int`: derivative order,
- `Symbol`: derivative backend tag,
- active scalar type.

# Notes
- This cache is intended for low-level derivative reuse inside the
  error-estimation workflow.
- Clear it with [`ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!`](@ref) when a fresh run is
  desired.
"""
const _NTH_DERIV_CACHE = Dict{Tuple, Any}()

"""
    _DERIV_JET_CACHE::Dict{Tuple{Any,Float64,Int,Symbol},Vector{Float64}}

Global cache for derivative jets.

# Description
This cache stores vectors of derivatives evaluated at a fixed point, typically
of the form

```julia
[f(x), f'(x), f''(x), ...]
```

up to a requested maximum order. Reusing a previously computed jet is often
more efficient than recomputing each derivative separately. The cache key also
tracks the active scalar type used for jet construction.

# Key structure
Each key has the form:

- callable identity,
- evaluation point in the active scalar type,
- `Int`: maximum derivative order,
- `Symbol`: derivative backend tag,
- active scalar type.

# Notes
- This cache is especially useful for Taylor-series or automatic-differentiation
  based derivative backends.
- Clear it with [`ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!`](@ref) when needed.
"""
const _DERIV_JET_CACHE = Dict{Tuple, AbstractVector}()

include("AutoDerivative/AutoDerivative.jl")
include("ErrorNewtonCotes/ErrorNewtonCotes.jl")
include("ErrorGauss/ErrorGauss.jl")
include("ErrorBSpline/ErrorBSpline.jl")
include("ErrorDispatch/ErrorDispatch.jl")

using .AutoDerivative
using .ErrorNewtonCotes
using .ErrorGauss
using .ErrorBSpline
using .ErrorDispatch

end  # module ErrorEstimate
