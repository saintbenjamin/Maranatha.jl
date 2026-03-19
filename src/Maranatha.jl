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
continuum extrapolation** on hyperrectangular domains

```math
\\prod_{i=1}^n [a_i, b_i]
```

which includes the special case of hypercubes ``[a,b]^n``.

Rather than stochastic sampling approaches (e.g. Monte Carlo or VEGAS),
Maranatha focuses on deterministic multi-resolution quadrature and
model-based continuum extrapolation to the limit ``h → 0``
using least-``\\chi^2`` fitting.

The framework follows a **pipeline-oriented workflow**:

1. deterministic quadrature evaluation
2. derivative-informed error-scale estimation
3. least-``\\chi^2`` extrapolation to ``h → 0``
4. convergence visualization

This design allows controlled numerical studies where convergence behavior
is explicitly modeled rather than statistically inferred.

The integration domain may be specified either as a scalar interval
(common bounds for all axes) or as axis-wise endpoints, allowing
general rectangular domains.

## Core components

The package is internally divided into independent submodules:

- [`Maranatha.Quadrature`](@ref)  
  Multi-dimensional tensor-product quadrature with rule dispatch
  (Newton–Cotes, Gauss-family, B-spline).

  Supports both uniform hypercubes and axis-wise rectangular domains.

  Supports multiple execution backends including
  serial evaluation, threaded subgrid partitioning,
  and CUDA-based GPU acceleration.

- [`Maranatha.ErrorEstimate`](@ref)  
  Unified error-scale modeling layer supporting both derivative-based
  residual models and refinement-based coarse-vs-refined estimators.

- [`Maranatha.LeastChiSquareFit`](@ref)  
  Weighted least-``χ²`` extrapolation routines for estimating the
  ``h → 0`` continuum limit.

- [`Maranatha.Documentation.PlotTools`](@ref)  
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

- [`plot_datapoints_result`](@ref)  
  Plot raw convergence datapoints before fitting.

- [`plot_quadrature_coverage_1d`](@ref)  
  Visualize one-dimensional quadrature structure for a selected rule.

- [`load_datapoint_results`](@ref)  
  Load a saved convergence dataset from disk.

- [`merge_datapoint_result_files`](@ref)  
  Merge multiple saved datapoint-result files into one.

- [`drop_nsamples_from_file`](@ref)  
  Remove selected subdivision counts from a saved result file.

- [`run_wizard`](@ref)  
  Launch the interactive TOML-configuration wizard.

## Typical workflow

```julia
using Maranatha

run_result = run_Maranatha("config.toml")

fit_result = least_chi_square_fit(run_result)

print_fit_result(fit_result)

plot_convergence_result(run_result, fit_result)
```

The integration bounds supplied to `run_Maranatha` may be scalars
(for ``[a,b]^n``) or tuples/vectors specifying per-axis limits.

For detailed documentation, tutorials, and complete workflow examples,
please refer to the project documentation site or the example
Jupyter notebooks located in the `ipynb/` directory of this repository.
"""
module Maranatha

using DoubleFloats
import Printf
import Statistics
import LinearAlgebra
# import Diffractor
import TOML

include("Utils/Utils.jl")
include("Quadrature/Quadrature.jl")
include("ErrorEstimate/ErrorEstimate.jl")
include("LeastChiSquareFit/LeastChiSquareFit.jl")
include("Integrands/Integrands.jl")
include("Runner/Runner.jl")
include("Documentation/Documentation.jl")

using .Utils
using .Quadrature
using .ErrorEstimate
using .LeastChiSquareFit
using .Integrands
using .Runner
using .Documentation

import .Runner: run_Maranatha
export run_Maranatha

import .LeastChiSquareFit: least_chi_square_fit, print_fit_result
export least_chi_square_fit, print_fit_result

import .Documentation.PlotTools: plot_convergence_result, plot_datapoints_result, plot_quadrature_coverage_1d
export plot_convergence_result, plot_datapoints_result, plot_quadrature_coverage_1d

import .Utils.MaranathaIO: load_datapoint_results, merge_datapoint_result_files, drop_nsamples_from_file
export load_datapoint_results, merge_datapoint_result_files, drop_nsamples_from_file

import .Utils.Wizard: run_wizard
export run_wizard

import .Documentation.Reporter: write_convergence_summary, write_convergence_internal_note, write_convergence_summary_datapoints, write_convergence_internal_note_datapoints
export write_convergence_summary, write_convergence_internal_note, write_convergence_summary_datapoints, write_convergence_internal_note_datapoints

end  # module Maranatha