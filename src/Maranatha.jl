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


# Examples

The example below demonstrates the standard three-step workflow:
generate a convergence dataset, perform the downstream fit, and plot the resulting convergence curve.

```julia
using Maranatha

f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

print_fit_result(fit)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```

If a long convergence study is executed in multiple partial runs
(for example, due to wall-time limits or interrupted sessions),
the saved `.jld2` files can later be merged with
[`Utils.MaranathaIO.merge_datapoint_result_files`](@ref)
or [`Utils.MaranathaIO.merge_datapoint_results`](@ref),
and the merged result can be passed to
[`LeastChiSquareFit.least_chi_square_fit`](@ref)
in exactly the same way as a single-run result.

```julia
using Maranatha

merged_path = merge_datapoint_result_files(
    "result_part1.jld2",
    "result_part2.jld2",
    "result_part3.jld2";
    output_path = "result_merged.jld2",
    write_summary = true,
)

merged = load_datapoint_results(merged_path)

fit = least_chi_square_fit(
    merged.a,
    merged.b,
    merged.h,
    merged.avg,
    merged.err,
    merged.rule,
    merged.boundary;
    nterms = merged.fit_terms,
    ff_shift = merged.ff_shift,
    nerr_terms = merged.nerr_terms
)
```

The merged output path may also be generated automatically from the
actual subdivision counts present in the merged result:

```julia
using Maranatha

merged_path = merge_datapoint_result_files(
    "result_part1.jld2",
    "result_part2.jld2",
    "result_part3.jld2";
    write_summary = true,
    output_dir = ".",
    name_prefix = "merged"
)

merged = load_datapoint_results(merged_path)
```

Then the output filename is automatically constructed as

```julia
result_merged_\$(rule)_\$(boundary)_N_2_3_4_5_6_7.jld2
```

Selected subdivision counts can also be removed from an existing result
file before fitting. This is useful when very coarse resolutions are
considered unreliable or visually inconsistent with the main trend.

```julia
using Maranatha

filtered_path = drop_nsamples_from_file(
    "result_full.jld2",
    [2, 3];
    write_summary = true,
    output_dir = ".",
    name_prefix = "filtered"
)
```

Then the returned `filtered_path` points to a new file containing the same result data but with the specified `N` values removed, so that only the remaining resolutions are included in the downstream fit.

```julia
[2,3,4,5,6,7] -> [4,5,6,7]
```

You may then load the filtered result and pass it to the fitting routine as usual:

```julia
filtered = load_datapoint_results(filtered_path)

fit = least_chi_square_fit(
    filtered.a,
    filtered.b,
    filtered.h,
    filtered.avg,
    filtered.err,
    filtered.rule,
    filtered.boundary;
    nterms = filtered.fit_terms,
    ff_shift = filtered.ff_shift,
    nerr_terms = filtered.nerr_terms
)
```

Before fitting, it can be useful to inspect only the raw datapoints in a
chosen `h^p` coordinate in order to check apparent linearity, oscillation,
or resolution-dependent irregularities.

```julia
using Maranatha

plot_datapoints_result(
    "merged_test",
    merged.h,
    merged.avg,
    merged.err;
    h_power = 4,
    xscale = :linear,
    yscale = :linear,
    ymode = :value,
    rule = merged.rule,
    boundary = merged.boundary,
)
```

A relative-difference diagnostic view can also be drawn on log-log axes
once a reference value is available:

```julia
using Maranatha

plot_datapoints_result(
    "merged_test",
    merged.h,
    merged.avg,
    merged.err;
    h_power = 4,
    xscale = :log,
    yscale = :log,
    ymode = :reldiff,
    reference_value = fit.estimate,
    rule = merged.rule,
    boundary = merged.boundary,
)
```

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