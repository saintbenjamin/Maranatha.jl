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

`Maranatha.Quadrature` provides the rule-dispatched numerical integration backend
for `Maranatha.jl`.

It supports:

- Exact composite Newton-Cotes construction
- Gauss-family rules (Legendre / Radau / Lobatto)
- B-spline-based quadrature
- Deterministic tensor-product extension to arbitrary dimension

The module is designed around reproducibility, analytical transparency,
and explicit control of rule and boundary behavior.

# Architecture

The implementation is organized into four submodules:

- [`Maranatha.Quadrature.NewtonCotes`](@ref): exact-rational composite
  Newton-Cotes construction
- [`Maranatha.Quadrature.Gauss`](@ref): Gauss-Legendre / Radau / Lobatto rules
- [`Maranatha.Quadrature.BSpline`](@ref): B-spline interpolation and smoothing
  quadrature
- [`Maranatha.Quadrature.QuadratureDispatch`](@ref): unified dispatch and
  tensor-product evaluation routines

# Multidimensional Strategy

All multidimensional integration is performed by explicit tensor-product
construction.

For dimension `d`:
```math
\\sum_{i_1 , i_2 , \\ldots , i_d} w_{i_1} w_{i_2} \\ldots w_{i_d} f\\left( x_{i_1} , x_{i_2} , \\ldots , x_{i_d} \\right)
```

The implementation uses explicit nested loops for ``d \\le 4`` and a general
multi-index traversal for arbitrary `dim`.

# Boundary Patterns

Boundary handling is centralized through:

    :LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX

These patterns determine rule-specific endpoint behavior and related boundary
semantics across the supported backends.

# Design Principles

- Exact arithmetic where mathematically appropriate
- No hidden adaptivity
- No implicit rule mutation
- Deterministic floating-point accumulation
- Explicit tensor-product semantics
- Strict validation of rule constraints

# Public API

Primary entry point:

- [`QuadratureDispatch.quadrature`](@ref)`(...)`

# Notes

- This module does not implement adaptive quadrature.
- No parallelism is applied at this layer.
- All rule-specific constraints are enforced internally.
"""
module Quadrature

import ..LinearAlgebra

import ..Utils.JobLoggerTools

include("QuadratureUtils.jl")
include("NewtonCotes.jl")
include("Gauss.jl")
include("BSpline.jl")
include("QuadratureNodes.jl")
include("QuadratureDispatch.jl")

using .QuadratureUtils
using .NewtonCotes
using .Gauss
using .BSpline
using .QuadratureNodes
using .QuadratureDispatch

end  # module Quadrature