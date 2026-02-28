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

`Maranatha.jl` is a modular Newton-Cotes-based toolkit for **multi-resolution
numerical integration**, **error-scale modeling**, and **``\\chi^2``-based convergence
extrapolation** on hypercube domains ``\\left[ a, b \\right]^n`` where ``n`` is the (spacetime) dimensionality.

Rather than exposing individual quadrature implementations directly,
`Maranatha.jl` is designed around a **pipeline-oriented workflow**:

1. integration
2. error estimation
3. ``h \\to 0`` extrapolation via least ``\\chi^2`` fitting
4. visualization using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).

The internal structure is intentionally split into small, independent modules
so that numerical components, logging, plotting, and preset integrands can
evolve without tightly coupling the codebase.

# Architecture overview

## Integration layer
[`Maranatha.Integrate`](@ref) module provides a unified front-end for tensor-product Newton-Cotes
quadrature in arbitrary dimensions. The concrete rule implementations are
kept internal and are not part of the public API surface.

## Error modeling
The [`Maranatha.ErrorEstimator`](@ref) module supplies lightweight derivative-based error
(scale) estimators that follow a tensor-product philosophy across dimensions.
For selected endpoint-free rules, boundary-difference leading-term models are
used to improve ``\\chi^2`` stability.

This estimator is **not rigorous truncation bound**; it is designed to
produce consistent scaling weights for least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation.

## Least ``\\chi^2`` fitting
[`Maranatha.LeastChiSquareFit`](@ref) submodule performs least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation using
a rule-dependent model
```math
I(h) = I_0 + C_1 \\, h^p + C_2 \\, h^{p+2} + ...
```
and returns parameter covariance, enabling uncertainty propagation into plots.

## Integrand system
The [`Maranatha.Integrands`](@ref) submodule implements a registry-based preset system that allows
named integrands to be constructed via factories while still accepting plain
`Julia` callables (functions, closures, callable structs).

This design keeps user-facing workflows simple without sacrificing flexibility.

## Execution layer
[`Maranatha.Runner.run_Maranatha`](@ref) is the main orchestration entry point.  
It performs:

1) multi-resolution integration,
2) error-scale estimation,
3) least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation,
4) formatted reporting of results.

Users typically interact only with this high-level interface.

## Plotting utilities
[`Maranatha.PlotTools.plot_convergence_result`](@ref) generates publication-style convergence
figures using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering. The shaded band represents the
full covariance-propagated ``1 \\, \\sigma`` uncertainty of the fitted model.

## Logging
All runtime diagnostics are handled by [`Maranatha.JobLoggerTools`](@ref), which provides
timestamped logging, stage delimiters, and timing macros used consistently
throughout the pipeline.

# Public API
The top-level Maranatha namespace re-exports a minimal set of entry points:

- [`Maranatha.Runner.run_Maranatha`](@ref)
    Perform numerical integration, error estimation, and least ``\\chi^2`` fitting for ``h \\to 0`` extrapolation.

- [`Maranatha.PlotTools.plot_convergence_result`](@ref)
    Visualize convergence behavior and fitted uncertainty bands.

Internal submodules remain accessible but are not required for normal usage.

# Dimensionality
Dimension-specific error estimators exist for low dimensions, along with a
generalized ``n``-dimensional implementation following the same tensor-product philosophy.
Because tensor enumeration scales rapidly with dimension, higher-dimensional
usage is primarily intended for controlled numerical studies.

# Design goals
- Pipeline-oriented structure rather than rule-centric APIs.
- Strict separation between numerical core, orchestration, and visualization.
- Reproducible floating-point behavior through preserved loop ordering.
- Minimal public API surface with extensible internal modules.
"""
module Maranatha

include("log/JobLoggerTools.jl")
using .JobLoggerTools

include("log/AvgErrFormatter.jl")
using .AvgErrFormatter

# ============================================================
# Integration dispatcher / high-level interface
#
# `Integrate.jl` typically defines a unified front-end that
# selects a specific quadrature rule depending on user options.
# ============================================================
include("generator/Integrate.jl")
using .Integrate

# ============================================================
# Error estimation and fitting utilities
#
# These modules provide:
#   - analytic / model-based error estimators
#   - convergence diagnostics for fitting pipelines
# ============================================================

include("error/ErrorEstimator.jl")
using .ErrorEstimator

include("fit/LeastChiSquareFit.jl")
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
include("integrands/Integrands.jl")
using .Integrands

# --- Raw integrand implementation (dependency of presets) ---
include("integrands/F0000GammaEminus1.jl")
using .F0000GammaEminus1

# --- Preset wrappers (user-facing integrands) ---
include("integrands/F0000.jl")
using .F0000Preset

# Register built-in presets
F0000Preset.__register_F0000_integrand__()

include("integrands/Z_q.jl")
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

include("controller/Runner.jl")
using .Runner

# ============================================================
# Plotting utilities
#
# PlotTools contains visualization helpers used for convergence
# diagnostics and presentation-quality output. These tools are
# kept separate from the numerical core so that plotting can
# evolve independently without affecting the integration logic.
# ============================================================

include("figs/PlotTools.jl")
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