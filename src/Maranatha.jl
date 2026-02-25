# ============================================================================
# src/Maranatha.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Maranatha

include("log/JobLoggerTools.jl")
using .JobLoggerTools

# ============================================================
# Numerical integration rules (Newton–Cotes family)
#
# Each rule is implemented as an independent submodule under
# `rules/`, and is brought into the namespace here so that the
# top-level Maranatha API can access them uniformly.
# ============================================================

# --- Closed rules (endpoint-evaluating) ---
include("rules/legacy/Simpson13Rule.jl")   # Composite Simpson 1/3 rule
include("rules/legacy/Simpson38Rule.jl")   # Composite Simpson 3/8 rule
include("rules/legacy/BodeRule.jl")        # Composite Bode/Boole rule (5-point closed NC)

using .Simpson13Rule
using .Simpson38Rule
using .BodeRule

# --- Globally-open rules (endpoint-free variants) ---
# These versions avoid evaluating f(a) and f(b), typically via
# interior stencils or endpoint elimination constructions.
include("rules/legacy/Simpson13Rule_MinOpen_MaxOpen.jl")
include("rules/legacy/Simpson38Rule_MinOpen_MaxOpen.jl")
include("rules/legacy/BodeRule_MinOpen_MaxOpen.jl")

using .Simpson13Rule_MinOpen_MaxOpen
using .Simpson38Rule_MinOpen_MaxOpen
using .BodeRule_MinOpen_MaxOpen

# ============================================================
# Integration dispatcher / high-level interface
#
# `Integrate.jl` typically defines a unified front-end that
# selects a specific quadrature rule depending on user options.
# ============================================================
include("rules/Integrate.jl")

using .Integrate

# ============================================================
# Error estimation and fitting utilities
#
# These modules provide:
#   - analytic / model-based error estimators
#   - Richardson extrapolation tools
#   - formatting utilities for averaged error output
#   - convergence diagnostics for fitting pipelines
# ============================================================

include("error/ErrorEstimator.jl")
include("error/RichardsonError.jl")
include("fit/AvgErrFormatter.jl")
include("fit/FitConvergence.jl")

using .ErrorEstimator
using .RichardsonError
using .AvgErrFormatter
using .FitConvergence

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