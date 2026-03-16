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

### 1. [`Maranatha.ErrorEstimate.ErrorNewtonCotes`](@ref)

Exact-rational residual extraction for Newton-Cotes rules.

### 2. [`Maranatha.ErrorEstimate.ErrorGauss`](@ref)

Float64 residual extraction for Gauss-family rules.

### 3. [`Maranatha.ErrorEstimate.ErrorBSpline`](@ref)

Float64 residual extraction for B-spline quadrature rules.

### 4. [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)

Unified dispatch layer for residual extraction, derivative probing, and
dimension-specific / dimension-generic residual-based error estimators.

## Refinement backends

### 5. [`Maranatha.ErrorEstimate.ErrorNewtonCotesRefine`](@ref)

Refinement-difference error estimation for Newton-Cotes rules.

### 6. [`Maranatha.ErrorEstimate.ErrorGaussRefine`](@ref)

Refinement-difference error estimation for Gauss-family rules.

### 7. [`Maranatha.ErrorEstimate.ErrorBSplineRefine`](@ref)

Refinement-difference error estimation for B-spline quadrature rules.

### 8. [`Maranatha.ErrorEstimate.ErrorDispatchRefine`](@ref)

Unified dispatch layer for rule-family refinement-based error estimation.

------------------------------------------------------------------------

# Derivative policy

For the residual-based branch, high-order derivatives are computed through the
shared wrapper [`ErrorDispatch.nth_derivative`](@ref), which uses a safe backend
policy and reports non-finite derivative events with contextual metadata.

All derivative probes in the residual-based branch are evaluated at the
physical midpoint.

The refinement-based branch does not use derivative probes; instead, it
constructs an error estimate directly from the difference between coarse and
refined quadrature evaluations.

------------------------------------------------------------------------

# Public API

Primary entry points include:

- [`ErrorDispatch.error_estimate`](@ref)
- [`ErrorDispatch.error_estimate_jet`](@ref)
- [`ErrorDispatchRefine.error_estimate_refine`](@ref)

These interfaces support dimension-specific and generic multidimensional
dispatch, depending on the selected backend.

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
import ..TaylorSeries
import ..Enzyme
import ..ForwardDiff
import ..FastDifferentiation
# import ..Diffractor

import ..Utils.JobLoggerTools
import ..Quadrature

include("ErrorNewtonCotes.jl")
include("ErrorNewtonCotesRefine.jl")
include("ErrorGauss.jl")
include("ErrorGaussRefine.jl")
include("ErrorBSpline.jl")
include("ErrorBSplineRefine.jl")
include("ErrorDispatch.jl")
include("ErrorDispatchRefine.jl")


using .ErrorNewtonCotes
using .ErrorNewtonCotesRefine
using .ErrorGauss
using .ErrorGaussRefine
using .ErrorBSpline
using .ErrorBSplineRefine
using .ErrorDispatch
using .ErrorDispatchRefine

end  # module ErrorEstimate