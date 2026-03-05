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
so that numerical components, logging, plotting, and preset integrands can
evolve without tightly coupling the codebase.

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
a *residual-informed exponent basis* derived from a residual-informed exponent basis dispatched by rule family (via the error-model backend).

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

[`Maranatha.Runner.run_Maranatha`](@ref) is the main orchestration entry point.
It performs:

1. multi-resolution quadrature,
2. error-scale estimation (optionally summing LO + NLO + ... residual terms via `nerr_terms`),
3. least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation (optionally shifting the fit basis via `ff_shift`),
4. formatted reporting of results.

Users typically interact only with this high-level interface.

## Plotting utilities

[`Maranatha.PlotTools.plot_convergence_result`](@ref) generates publication-style convergence
figures using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering. The shaded band represents the
full covariance-propagated ``1 \\, \\sigma`` uncertainty of the fitted model.
The plot reconstruction uses the exact exponent basis stored in the fit result
(e.g. `fit_result.powers`) together with the parameter covariance (e.g. `fit_result.cov`),
ensuring consistency with the fitted model, including any forward-shift (`ff_shift`) applied.
Convergence is visualized against ``h^{p}`` where ``p = \texttt{fit_result.powers[2]}``,
with model evaluation performed on ``h`` via ``h = x^{1/p}``.

## Logging

All runtime diagnostics are handled by [`Maranatha.Utils.JobLoggerTools`](@ref), which provides
timestamped logging, stage delimiters, and timing macros used consistently
throughout the pipeline.

# Public API

The top-level Maranatha namespace re-exports a minimal set of entry points:

* [`Maranatha.Runner.run_Maranatha`](@ref)
  Perform numerical integration, error estimation, and least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation.

* [`Maranatha.PlotTools.plot_convergence_result`](@ref)
  Visualize convergence behavior and fitted uncertainty bands.

Internal submodules remain accessible but are not required for normal usage.

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

using LinearAlgebra
using TaylorSeries
using Enzyme
using ForwardDiff
using FastDifferentiation
# using Diffractor

include("Utils/Utils.jl")
using .Utils

# ============================================================
# Integration dispatcher / high-level interface
#
# `Quadrature.jl` typically defines a unified front-end that
# selects a specific quadrature rule depending on user options.
# ============================================================
include("Quadrature/Quadrature.jl")
using .Quadrature

# ============================================================
# Error estimation and fitting utilities
#
# These modules provide:
#   - analytic / model-based error estimators
#   - convergence diagnostics for fitting pipelines
# ============================================================
include("ErrorEstimate/ErrorEstimate.jl")
using .ErrorEstimate

include("LeastChiSquareFit/LeastChiSquareFit.jl")
using .LeastChiSquareFit

# ============================================================
# Integrand system
#
# The integrand registry allows complex integrands to be exposed
# as user-friendly presets, while still supporting plain Julia
# functions entered directly in the REPL.
#
# Order matters:
#   1) Integrands registry
#   2) Raw integrand implementations (e.g. F0000GammaEminus1)
#   3) Preset wrappers that depend on them
# ============================================================

# --- Integrand registry core ---
include("Integrands/Integrands.jl")
using .Integrands

# --- Raw integrand implementation (dependency of presets) ---
include("Integrands/F0000GammaEminus1.jl")
using .F0000GammaEminus1

# --- Preset wrappers (user-facing integrands) ---
include("Integrands/F0000.jl")
using .F0000Preset

# Register built-in presets
F0000Preset.__register_F0000_integrand__()

include("Integrands/Z_q.jl")
using .Z_q

# ============================================================
# Controller / execution layer
#
# The Runner module provides the high-level orchestration layer
# that connects the quadrature rules, error estimators, and
# convergence fitting logic into a single workflow entry point.
#
# Users are NOT expected to interact with Runner directly.
# Instead, selected functions are re-exported at the top level
# of the Maranatha namespace for convenience and API clarity.
#
# Design principle:
#   - Internal modules remain modular and independent.
#   - The top-level Maranatha API exposes a minimal, stable
#     surface consisting of user-facing entry points only.
# ============================================================

include("Runner/Runner.jl")
using .Runner

# ============================================================
# Plotting utilities
#
# PlotTools contains visualization helpers used for convergence
# diagnostics and presentation-quality output. These tools are
# kept separate from the numerical core so that plotting can
# evolve independently without affecting the integration logic.
# ============================================================

include("PlotTools/PlotTools.jl")
using .PlotTools

# ============================================================
# Public API re-exports
#
# The following aliases expose selected internal functionality
# as part of the public Maranatha interface.
#
# This avoids requiring users to call:
#     Maranatha.Runner.run_Maranatha(...)
# or:
#     Maranatha.PlotTools.plot_convergence_result(...)
#
# Instead, users can simply write:
#     using Maranatha
#     run_Maranatha(...)
#     plot_convergence_result(...)
#
# The `const` alias preserves performance and ensures that the
# binding remains stable across the module lifetime.
# ============================================================

# Main execution entry point
const run_Maranatha = Runner.run_Maranatha
export run_Maranatha

# Convergence plotting helper
const plot_convergence_result = PlotTools.plot_convergence_result
export plot_convergence_result

end  # module Maranatha