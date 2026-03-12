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

Residual-based truncation-error modeling engine used by `Maranatha.jl`.

# Overview

`Maranatha.ErrorEstimate` provides the asymptotic truncation-error modeling layer
for the structured tensor-product quadrature backends in `Maranatha.jl`.

It implements a unified framework that:

- extracts leading nonzero midpoint-centered residual moments of composite quadrature rules, and
- combines them with high-order derivative probes at the physical midpoint to form an
  axis-separable truncation-error model.

This module is intended for:

- fit stabilization,
- error-scale diagnostics,
- automatic leading-order detection from residual structure.

It is not intended to provide strict error bounds.

------------------------------------------------------------------------

# Core idea

1. Build a composite quadrature rule on a dimensionless tiling grid.
2. Detect leading nonzero residual moment orders from the rule structure.
3. Convert those residuals into factorial-scaled coefficients.
4. Combine them with midpoint derivative probes to construct a truncation-error model.

The model is axis-separable by construction; mixed-derivative contributions are
treated as higher-order effects and are intentionally omitted.

------------------------------------------------------------------------

# Architecture

The module is structured into three residual backends plus a unified dispatch layer:

## 1. [`Maranatha.ErrorEstimate.ErrorNewtonCotes`](@ref)

Exact-rational residual extraction for Newton-Cotes rules.

## 2. [`Maranatha.ErrorEstimate.ErrorGauss`](@ref)

Float64 residual extraction for Gauss-family rules.

## 3. [`Maranatha.ErrorEstimate.ErrorBSpline`](@ref)

Float64 residual extraction for B-spline quadrature rules.

## 4. [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)

Unified dispatch layer for residual extraction, derivative probing, and
dimension-specific / dimension-generic error estimators.

------------------------------------------------------------------------

# Derivative policy

High-order derivatives are computed through the shared wrapper
[`ErrorDispatch.nth_derivative`](@ref), which uses a safe backend policy and
reports non-finite derivative events with contextual metadata.

All derivative probes are evaluated at the physical midpoint.

------------------------------------------------------------------------

# Public API

Primary entry points:

- [`ErrorDispatch.error_estimate`](@ref)
- [`ErrorDispatch.error_estimate_threads`](@ref)

Both support multiple residual terms via `nerr_terms` and dispatch to
dimension-specific or generic multidimensional implementations.

------------------------------------------------------------------------

# Notes

- This module models truncation error for structured composite rules; it does not perform fitting itself.
- Numerical decisions in `Float64` residual detection are controlled by backend tolerances.
- Computational cost grows rapidly with dimension in generic multidimensional estimators.
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
include("ErrorGauss.jl")
include("ErrorBSpline.jl")
include("ErrorDispatch.jl")

using .ErrorNewtonCotes
using .ErrorGauss
using .ErrorBSpline
using .ErrorDispatch

end  # module ErrorEstimate