# ============================================================================
# src/Quadrature/Quadrature.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Quadrature

Unified tensor-product quadrature engine used by `Maranatha.jl`.

# Overview

`Maranatha.Quadrature` provides a modular, rule-dispatched numerical integration
framework supporting:

- Exact composite Newton-Cotes (rational assembly)
- Gauss-family rules (Legendre / Radau / Lobatto)
- B-spline-based quadrature (interpolation & smoothing)
- Deterministic tensor-product extension to arbitrary dimension

The module is designed for:

- Reproducibility
- Analytical transparency
- Explicit control of boundary behavior
- Exact-weight construction where mathematically meaningful

It serves as the numerical integration backend for the
higher-level `Maranatha.jl` pipeline.

------------------------------------------------------------------------

# Architecture

The module is structured into four rule backends plus a dispatcher layer:

## 1. [`Maranatha.Quadrature.NewtonCotes`](@ref)

Exact-rational composite Newton-Cotes construction.

Features:

- `Rational{BigInt}` exact local moment matching
- Exact global `\\beta` coefficient assembly
- Composite boundary tiling validation
- `Float64` conversion only at final stage
- Process-local caching of assembled weights

Supported rule symbols:

    :newton_pK    (e.g., :newton_p3, :newton_p5, ...)

Boundary patterns:

    :LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX

------------------------------------------------------------------------

## 2. [`Maranatha.Quadrature.Gauss`](@ref)

Single-interval Gauss-family rules on ``[-1,1]``:

- Gauss-Legendre
- Gauss-Radau (left or right)
- Gauss-Lobatto

Composite repetition over uniform subintervals is supported.

Supported rule symbols:

    :gauss_pN

Boundary selects family:

    :LU_EXEX -> Legendre
    :LU_INEX -> Radau (left)
    :LU_EXIN -> Radau (right)
    :LU_ININ -> Lobatto

Nodes/weights are cached per `(n, boundary)` pair.

------------------------------------------------------------------------

## 3. [`Maranatha.Quadrature.BSpline`](@ref)

B-spline-based quadrature using:

- Uniform knot construction
- Greville abscissae nodes
- Exact basis integrals
- Interpolation mode
- Optional smoothing mode (Tikhonov second-difference penalty)

Supported rule symbols:

    :bspline_interp_pK -> interpolation
    :bspline_smooth_pK -> smoothing

Boundary controls endpoint clamping.

------------------------------------------------------------------------

## 4. [`Maranatha.Quadrature.QuadratureDispatch`](@ref)

Provides:

- [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref)
- Tensor-product evaluation routines:
    - [`Maranatha.Quadrature.QuadratureDispatch.quadrature_1d`](@ref)
    - [`Maranatha.Quadrature.QuadratureDispatch.quadrature_2d`](@ref)
    - [`Maranatha.Quadrature.QuadratureDispatch.quadrature_3d`](@ref)
    - [`Maranatha.Quadrature.QuadratureDispatch.quadrature_4d`](@ref)
    - [`Maranatha.Quadrature.QuadratureDispatch.quadrature_nd`](@ref)
- Unified front-end:

    - [`quadrature`](@ref Maranatha.Quadrature.QuadratureDispatch.quadrature)`(f, a, b, N, dim, rule, boundary)`

------------------------------------------------------------------------

# Multidimensional Strategy

All multidimensional integration is performed via explicit
tensor-product construction.

For dimension `d`:
```math
\\sum_{i_1 , i_2 , \\ldots , i_d} w_{i_1} w_{i_2} \\ldots w_{i_d} f\\left( x_{i_1} , x_{i_2} , \\ldots , x_{i_d} \\right)
```

## The implementation:

- Uses explicit nested loops for ``d \\le 4``
- Uses an odometer-style multi-index for general `dim`
- Skips zero-weight entries for efficiency
- Preserves deterministic accumulation ordering

Computational cost scales as:
```math
\\mathcal{O} \\left( \\text{length(xs)}^{\\texttt{dim}} \\right)
```

------------------------------------------------------------------------

# Boundary Patterns

Boundary handling is centralized via:

    :LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX

These patterns determine:

- Endpoint inclusion/exclusion for Newton-Cotes
- Family selection for Gauss rules
- Knot clamping behavior for B-splines

------------------------------------------------------------------------

# Design Principles

- Exact arithmetic where mathematically appropriate
- No hidden adaptivity
- No implicit rule mutation
- Deterministic floating-point accumulation
- Explicit tensor-product semantics
- Strict validation of rule constraints

------------------------------------------------------------------------

# Public API

Primary entry points:

- [`quadrature`](@ref Maranatha.Quadrature.QuadratureDispatch.quadrature)`(...)`
- [`get_quadrature_1d_nodes_weights`](@ref Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights)`(...)`

------------------------------------------------------------------------

# Notes

- This module does not implement adaptive quadrature.
- No parallelism is applied at this layer.
- All rule-specific constraints are enforced internally.
- Intended for structured research-grade numerical experiments,
  not black-box production integration.
"""
module Quadrature

using ..LinearAlgebra

using ..Utils.JobLoggerTools

export quadrature, get_quadrature_1d_nodes_weights

include("NewtonCotes.jl")
include("Gauss.jl")
include("BSpline.jl")
include("QuadratureDispatch.jl")

using .NewtonCotes
using .Gauss
using .BSpline
using .QuadratureDispatch

end  # module Quadrature