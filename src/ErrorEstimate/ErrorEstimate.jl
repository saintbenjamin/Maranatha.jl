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

`Maranatha.ErrorEstimate` provides the **asymptotic truncation-error modeling layer**
for the structured tensor-product quadrature backends in `Maranatha.jl`.

It implements a unified framework that:

- extracts leading nonzero **midpoint-centered residual moments** of composite quadrature rules, and
- combines them with **high-order derivative probes at the physical midpoint** to form an
  **axis-separable truncation-error model**.

This module is designed for:

- Fit stabilization (e.g., weighted extrapolation as ``h \\to 0``)
- Error-scale diagnostics (how error magnitudes *should* scale with ``h``)
- Automatic order detection (leading power selection from residual structure)

It is **not** intended to provide strict error bounds.

------------------------------------------------------------------------

# Core idea

1. Build a composite quadrature rule on a dimensionless tiling grid
   ``u \\in [0, N_{\\texttt{sub}}]`` with midpoint shift ``c = \\dfrac{N_{\\texttt{sub}}}{2}``.
2. Detect the first nonzero residual moment orders ``k`` using either:
   - **exact rational arithmetic** (Newton-Cotes NS rules), or
   - **tolerance-based Float64 probing** (Gauss-family rules and B-spline rules).
3. Convert each detected residual into a factorial-scaled coefficient
   ``\\texttt{coeff}_k = \\dfrac{\\texttt{diff}_k}{k!}``.
4. Assemble a truncation-error *model* by weighting midpoint derivatives of the integrand:
   ```math
   E \\approx \\sum_{i=1}^{n_{\\texttt{err}}} \\texttt{coeff}_{k_i}\\, h^{k_i+1}
   \\sum_{\\mu=1}^{\\texttt{dim}} I_{\\mu}^{(k_i)} \\,,
   ```
   where each ``I_{\\mu}^{(k)}`` is a cross-axis integral of a single-axis ``k``-th derivative,
   computed numerically with the same 1D quadrature nodes/weights.

The model is **axis-separable** by construction (sum of single-axis error operators);
mixed-derivative contributions are higher order and intentionally omitted.

------------------------------------------------------------------------

# Architecture

The module is structured into three residual backends plus a unified dispatch layer:

## 1. [`Maranatha.ErrorEstimate.ErrorNewtonCotes`](@ref)

Exact-rational residual extraction for NS-style composite Newton-Cotes rules (`:newton_pK`).

Features:

- Exact composite coefficient-vector assembly and exact moment tests
- Exact detection of nonzero residual moments (`diff != 0` in rational arithmetic)
- Residual coefficients returned as exact rationals (converted to `Float64` only at the end)

------------------------------------------------------------------------

## 2. [`Maranatha.ErrorEstimate.ErrorGauss`](@ref)

Float64 residual detection for composite Gauss-family rules (`:gauss_pK`).

Features:

- Dimensionless composite grids produced by the Gauss backend
- Residual nonzero detection via tolerance tests:
  ```math
  \\left| \\texttt{diff}_k \\right| > \\texttt{tol\\_abs} + \\texttt{tol\\_rel} \\, \\left| \\texttt{exact} \\right|
  ```
- Coefficients returned directly in `Float64`

------------------------------------------------------------------------

## 3. [`Maranatha.ErrorEstimate.ErrorBSpline`](@ref)

Float64 residual detection for composite B-spline quadrature rules (`:bspline_interp_pK`, `:bspline_smooth_pK`).

Features:

- Residual extraction based on midpoint-shifted monomial probing on the composite grid
- Tolerance-controlled nonzero detection (mirrors the Gauss-family policy)
- Supports both interpolation and smoothing B-spline rule families

------------------------------------------------------------------------

## 4. [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)

Unified dispatch layer that:

- Normalizes residual backends into a common interface:
  `(ks, coeffs_float, center)`
- Provides dimension-specific estimators:
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_1d`](@ref), 
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_2d`](@ref), 
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_3d`](@ref), 
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_4d`](@ref)
- Provides dimension-generic estimators:
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_nd`](@ref) (and threaded variants)
- Exposes threaded implementations using [`Base.Threads`](https://docs.julialang.org/en/v1/base/multi-threading/)

This is the layer that the public API calls.

------------------------------------------------------------------------

# Derivative backends and safety policy

High-order derivatives are computed by a shared safe wrapper, [`Maranatha.ErrorEstimate.ErrorDispatch.nth_derivative`](@ref):

- Primary backend: [`ForwardDiff.derivative`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.derivative) (closure-chain for higher orders)
- Fallback backend: [`TaylorSeries.Taylor1`](https://juliadiff.org/TaylorSeries.jl/stable/api/#TaylorSeries.Taylor1-Union{Tuple{T},%20Tuple{Type{T},%20Int64}}%20where%20T%3C:Number) expansion (single-pass coefficient extraction)
- If both produce non-finite values: a fatal error is raised via `JobLoggerTools`

All derivative probes are evaluated at the physical midpoint (e.g., ``\\displaystyle{\\bar{x}=\\dfrac{a+b}{2}}``),
and warnings/errors include contextual metadata (`h`, `rule`, `N`, `dim`, axis tags, stage tags)
to diagnose non-finite derivative events.

------------------------------------------------------------------------

# Centering convention

Residuals are currently defined using the midpoint shift (center tag `:mid`).
The dispatch interface returns the center symbol explicitly so future extensions
can support alternative centers without breaking downstream signatures.

------------------------------------------------------------------------

# Public API

Primary entry points:

- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)`(f, a, b, N, dim, rule, boundary; nerr_terms=1)`
- [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_threads`](@ref)`(f, a, b, N, dim, rule, boundary; nerr_terms=1)`

Both entry points:

- Support collecting multiple residual terms via `nerr_terms` (LO, LO+NLO, ...)
- Dispatch to `dim = 1,2,3,4` specialized implementations and `dim >= 5` generic multidimensional logic
- Use the same residual-coefficient extraction pipeline via [`Maranatha.ErrorEstimate.ErrorDispatch`](@ref)

------------------------------------------------------------------------

# Notes

- This module models truncation error for structured composite rules; it does not perform fitting itself.
- Numerical decisions in `Float64` residual detection are controlled by tolerance parameters in the backends.
- Computational cost grows rapidly with dimension in the generic multidimensional estimator due to explicit tensor-product enumeration.
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

export error_estimate, error_estimate_threads

end  # module ErrorEstimate