# ============================================================================
# src/Maranatha.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Maranatha

`Maranatha.jl` is a numerical framework for **deterministic quadrature-based
continuum extrapolation** on hypercube domains ``[a,b]^n``.

Instead of relying on stochastic sampling (e.g. Monte Carlo or VEGAS),
Maranatha evaluates integrals using deterministic quadrature rules across
multiple resolutions and extrapolates the result to the
continuum limit ``h → 0`` using least-``χ²`` fitting.

The framework follows a **pipeline-oriented workflow**:

1. deterministic quadrature evaluation
2. derivative-informed error-scale estimation
3. least-``χ²`` extrapolation to ``h → 0``
4. convergence visualization

This design allows controlled numerical studies where convergence behavior
is explicitly modeled rather than statistically inferred.

## Core components

The package is internally divided into independent submodules:

- [`Maranatha.Quadrature`](@ref)  
  Multi-dimensional tensor-product quadrature with rule dispatch
  (Newton–Cotes, Gauss-family, B-spline).

- [`Maranatha.ErrorEstimate`](@ref)  
  Derivative-based error-scale models derived from rule-family residual
  expansions.

- [`Maranatha.LeastChiSquareFit`](@ref)  
  Weighted least-``χ²`` extrapolation routines for estimating the
  ``h → 0`` continuum limit.

- [`Maranatha.PlotTools`](@ref)  
  Visualization utilities for convergence behavior and quadrature structure.

- [`Maranatha.Integrands`](@ref)  
  Registry-based system for reusable preset integrands.

- [`Maranatha.Utils`](@ref)  
  Shared infrastructure including logging utilities and helper tools.

## Public API

The top-level namespace re-exports a minimal set of entry points:

- [`run_Maranatha`](@ref)  
  Execute multi-resolution quadrature and build a convergence dataset.

- [`least_chi_square_fit`](@ref)  
  Perform weighted least ``χ²`` fitting for ``h → 0`` extrapolation.

- [`print_fit_result`](@ref)  
  Print formatted summaries of fitted parameters and diagnostics.

- [`plot_convergence_result`](@ref)  
  Generate convergence plots with fitted uncertainty bands.

## Typical workflow

```julia
using Maranatha

run_result = run_Maranatha("config.toml")

fit_result = least_chi_square_fit(run_result)

print_fit_result(fit_result)

plot_convergence_result(run_result, fit_result)
```

For detailed documentation, tutorials, and complete workflow examples,
please refer to the project documentation site or the example
Jupyter notebooks located in the `ipynb/` directory of this repository.
"""
module Maranatha

import Printf
import Dates
import Statistics
import LinearAlgebra
import TaylorSeries
import Enzyme
import ForwardDiff
import FastDifferentiation
# import Diffractor
import TOML
import JLD2
import PyPlot

include("Utils/Utils.jl")
include("Quadrature/Quadrature.jl")
include("ErrorEstimate/ErrorEstimate.jl")
include("LeastChiSquareFit/LeastChiSquareFit.jl")
include("Integrands/Integrands.jl")
include("Runner/Runner.jl")
include("PlotTools/PlotTools.jl")

using .Utils
using .Quadrature
using .ErrorEstimate
using .LeastChiSquareFit
using .Integrands
using .Runner
using .PlotTools

import .Runner: run_Maranatha
export run_Maranatha

import .LeastChiSquareFit: least_chi_square_fit, print_fit_result
export least_chi_square_fit, print_fit_result

import .PlotTools: plot_convergence_result, plot_datapoints_result, plot_quadrature_coverage_1d
export plot_convergence_result, plot_datapoints_result, plot_quadrature_coverage_1d

import .Utils.MaranathaIO: load_datapoint_results, merge_datapoint_result_files, drop_nsamples_from_file
export load_datapoint_results, merge_datapoint_result_files, drop_nsamples_from_file

import .Utils.Wizard: run_wizard
export run_wizard

import .Utils.Reporter: write_convergence_summary, write_convergence_internal_note
export write_convergence_summary, write_convergence_internal_note

end  # module Maranatha