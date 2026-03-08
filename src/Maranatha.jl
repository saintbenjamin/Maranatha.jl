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

Maranatha.jl is a framework for deterministic quadrature-based
continuum extrapolation on hypercube domains ``[a,b]^n`` 
where ``n`` is the (spacetime) dimensionality.

It provides tools for multi-dimensional quadrature,
derivative-informed error-scale modeling, and least χ² fitting
for estimating the limit ``h \to 0``.

Many numerical integration tools rely on stochastic sampling
(e.g. Monte Carlo or VEGAS-style algorithms). While such methods
scale well in very high dimensions, they provide statistical
estimates whose uncertainty decreases only slowly with sampling.

`Maranatha.jl` instead focuses on deterministic quadrature rules
combined with resolution scaling and continuum extrapolation.
By evaluating the integral at multiple step sizes and fitting
the expected convergence behavior, the framework estimates the
limit ``h \\to 0`` together with a model-informed uncertainty.

This approach is particularly useful for controlled numerical
studies where convergence behavior is important and where
deterministic sampling structure can be exploited.

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

# Example workflow

The example below demonstrates a minimal end-to-end workflow
using a configuration file and a simple integrand definition.

First define a small integrand in a Julia source file.

Example integrand (`sample_1d.jl`)

```julia
integrand(x) = sin(x)
```

Next prepare a configuration file describing the integration
domain, sampling sequence, quadrature rule, and output options.

Configuration file (`sample_1d.toml`)

```toml
[integrand]
file = "sample_1d.jl"
name = "integrand"

[domain]
a = 0.0
b = 3.141592653589793
dim = 1

[sampling]
nsamples = [2, 3, 4, 5, 6, 7, 8, 9]

[quadrature]
rule = "gauss_p4"
boundary = "LU_EXEX"

[error]
err_method = "forwarddiff"
fit_terms = 4
nerr_terms = 3
ff_shift = 0

[execution]
use_threads = true

[output]
name_prefix = "1D"
save_path = "."
write_summary = true
save_file = true
```

Assume that `sample_1d.jl` and `sample_1d.toml` are located
in the current working directory.

The quadrature pipeline can then be executed using the
high-level runner, producing a convergence dataset 
across multiple quadrature resolutions.

```julia
using Maranatha

run_result = run_Maranatha("./sample_1d.toml")
```

Once the dataset has been generated, the continuum limit
``h \\to 0`` can be estimated by performing a least ``\\chi^2`` fit.

```julia
fit_result = least_chi_square_fit(
    run_result; 
    nterms=3, 
    ff_shift=0, 
    nerr_terms=2
)

print_fit_result(fit_result)
```

Finally, the convergence behavior and fitted uncertainty
can be visualized using the plotting utilities.

```julia
plot_convergence_result(
    run_result, 
    fit_result;
    name="Maranatha_test1",
    figs_dir=".",
    save_file=true
)
```

For more detailed examples and interactive demonstrations,
see the Jupyter notebooks in the `ipynb/` directory of this project.

These notebooks provide step-by-step tutorials covering the full
Maranatha workflow, including dataset generation, merging partial
runs, filtering datapoints, and convergence visualization.
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

end  # module Maranatha