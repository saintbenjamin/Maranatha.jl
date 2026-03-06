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

`Maranatha.jl` is a modular toolkit for **multi-dimensional quadrature**,
**error-scale modeling**, and **least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation**
on hypercube domains ``[a,b]^n`` where ``n`` is the (spacetime) dimensionality.

The quadrature layer supports multiple rule backends (Newton-Cotes, Gauss-family, and B-spline),
selected via rule dispatch.

`Maranatha.jl` is designed around a **pipeline-oriented workflow**:

1. quadrature
2. error estimation
3. ``h \\to 0`` extrapolation via least ``\\chi^2`` fitting
4. visualization using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).

The internal structure is intentionally split into small, independent submodules
so that numerical components, dataset generation, fitting, logging, plotting, and
preset integrands can evolve without tightly coupling the codebase.

In the current design, these pipeline stages are exposed as separate responsibilities:
the runner builds raw convergence data, the fitter performs ``h \\to 0`` extrapolation,
and plotting utilities visualize either the fitted result or the underlying quadrature behavior.

# Architecture overview

## Integration layer
[`Maranatha.Quadrature`](@ref) provides a unified front-end for tensor-product quadrature
in arbitrary dimensions, dispatching to multiple rule backends (Newton-Cotes / Gauss-family / B-spline).
The concrete rule implementations are
kept internal and are not part of the public API surface. The quadrature core uses an exact-moment / Taylor-expansion-based construction to support
general multi-point composite Newton-Cotes rules through a unified implementation (including
configurable endpoint openness via boundary patterns).

## Error modeling
The [`Maranatha.ErrorEstimate`](@ref) module supplies lightweight derivative-based error
(scale) estimators that follow a tensor-product philosophy across dimensions.
The estimator uses a lightweight derivative-based *error scale* model whose leading structure is
determined from a rule-family residual model (dispatched by `rule`),
using midpoint residual moments/terms derived from the underlying composite weights.

The estimator supports using one or more residual terms (LO, NLO, ...) via a term count parameter
(e.g. `nerr_terms` in the high-level runner), which sums multiple midpoint-residual contributions when requested.
This provides consistent ``h``-scaling weights for least-``\\chi^2`` fitting across dimensions.

This estimator is *not a rigorous truncation bound*; it is designed to
produce consistent scaling weights for least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation.

## Least ``\\chi^2`` fitting
[`Maranatha.LeastChiSquareFit`](@ref) submodule performs least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation using
a *residual-informed exponent basis* dispatched by rule family (via the error-model backend).

The fitted model is linear in its parameters:
```math
I(h) = \\sum_{\\texttt{i}=1}^{n} \\lambda_\\texttt{i} \\, h^{\\,\\texttt{powers[i]}}
```
where the exponent vector `powers` is determined during fitting and stored in the fit result
(e.g. `fit_result.powers`, with `powers[1] = 0` for the constant term).

The fitter may optionally apply a **fitting-function-shift** (e.g. `ff_shift`) when selecting residual-derived powers,
allowing the fit basis to skip a nominal leading order that is expected to vanish for the given integrand.
In such cases, the stored `fit_result.powers` reflects the shifted basis actually used in the fit.

The routine also returns the parameter covariance matrix, enabling covariance-propagated uncertainty
bands in convergence plots.

## Integrand system

The [`Maranatha.Integrands`](@ref) submodule implements a registry-based preset system that allows
named integrands to be constructed via factories while still accepting plain
`Julia` callables (functions, closures, callable structs).

This design keeps user-facing workflows simple without sacrificing flexibility.

## Execution layer

[`Maranatha.Runner.run_Maranatha`](@ref) is the main orchestration entry point for
building a raw convergence dataset.

It performs:

1. multi-resolution quadrature,
2. error-scale estimation (optionally summing LO + NLO + ... residual terms via `nerr_terms`),
3. collection and optional saving of the resulting convergence data.

The returned dataset is then intended to be passed to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
for downstream ``h \\to 0`` extrapolation.

Formatted reporting of fit results is handled separately by
[`Maranatha.LeastChiSquareFit.print_fit_result`](@ref).

Users will often begin with this high-level runner, but fitting and reporting are now
explicit downstream steps rather than responsibilities of the runner itself.

## Plotting utilities

[`Maranatha.PlotTools.plot_convergence_result`](@ref) generates publication-style convergence
figures using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering. The shaded band represents the
covariance-propagated ``1 \\, \\sigma`` uncertainty of the fitted model.
The plot reconstruction uses the exact exponent basis stored in the fit result
(e.g. `fit_result.powers`) together with the parameter covariance (e.g. `fit_result.cov`),
ensuring consistency with the fitted model, including any forward-shift (`ff_shift`) applied.
Convergence is visualized against ``h^{p}`` where ``p = \\texttt{fit\\_result.powers[2]}``,
with model evaluation performed on ``h`` via ``h = x^{1/p}``.

[`Maranatha.PlotTools.plot_quadrature_coverage_1d`](@ref) provides a complementary
pedagogical visualization of how a selected 1D quadrature rule samples and approximates
the integrand, including B-spline reconstruction and non-B-spline contribution views.

## Logging

All runtime diagnostics are handled by [`Maranatha.Utils.JobLoggerTools`](@ref), which provides
timestamped logging, stage delimiters, and timing macros used consistently
throughout the pipeline.

# Public API

The top-level Maranatha namespace re-exports a minimal set of entry points:

* [`Maranatha.Runner.run_Maranatha`](@ref)
  Perform numerical integration and error-scale estimation across multiple resolutions,
  returning a raw convergence dataset for later fitting.

* [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
  Perform weighted least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation from a raw convergence dataset.

* [`Maranatha.LeastChiSquareFit.print_fit_result`](@ref)
  Print a formatted summary of the fitted parameters, uncertainties, and ``\\chi^2`` diagnostics.

* [`Maranatha.PlotTools.plot_convergence_result`](@ref)
  Visualize convergence behavior and fitted uncertainty bands.

Internal submodules remain accessible but are not required for normal usage,
which typically proceeds through a small set of top-level dataset-generation,
fitting, reporting, and plotting entry points.

# Dimensionality

A generalized ``n``-dimensional implementation is provided following the same
tensor-product philosophy, with dimension-specific specializations available
for lower dimensions.
Because tensor enumeration scales rapidly with dimension, higher-dimensional
usage is primarily intended for controlled numerical studies.

# Design goals

* Pipeline-oriented structure rather than rule-centric APIs.
* Strict separation between numerical core, orchestration, and visualization.
* Reproducible floating-point behavior through preserved loop ordering.
* Minimal public API surface with extensible internal modules.
* Unified general multi-point Newton-Cotes support via Taylor-expansion/moment-based rule construction.
"""
module Maranatha

using Printf
using Statistics
using LinearAlgebra
using TaylorSeries
using Enzyme
using ForwardDiff
using FastDifferentiation
# using Diffractor
using TOML
using JLD2
using PyPlot

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

const run_Maranatha = Runner.run_Maranatha
export run_Maranatha

const least_chi_square_fit = LeastChiSquareFit.least_chi_square_fit
export least_chi_square_fit
const print_fit_result = LeastChiSquareFit.print_fit_result
export print_fit_result

const plot_convergence_result = PlotTools.plot_convergence_result
export plot_convergence_result
const plot_quadrature_coverage_1d = PlotTools.plot_quadrature_coverage_1d
export plot_quadrature_coverage_1d

const load_datapoint_results = Maranatha.Utils.MaranathaIO.load_datapoint_results
export load_datapoint_results

end  # module Maranatha