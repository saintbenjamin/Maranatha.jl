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

# ============================================================
# Numerical integration rules (Newton–Cotes family)
#
# Each rule is implemented as an independent submodule under
# `rules/`, and is brought into the namespace here so that the
# top-level Maranatha API can access them uniformly.
# ============================================================

# --- Closed rules (endpoint-evaluating) ---
include("rules/Simpson13Rule.jl")   # Composite Simpson 1/3 rule
include("rules/Simpson38Rule.jl")   # Composite Simpson 3/8 rule
include("rules/BodeRule.jl")        # Composite Bode/Boole rule (5-point closed NC)

using .Simpson13Rule
using .Simpson38Rule
using .BodeRule

# --- Globally-open rules (endpoint-free variants) ---
# These versions avoid evaluating f(a) and f(b), typically via
# interior stencils or endpoint elimination constructions.
include("rules/Simpson13Rule_MinOpen_MaxOpen.jl")
include("rules/Simpson38Rule_MinOpen_MaxOpen.jl")
include("rules/BodeRule_MinOpen_MaxOpen.jl")

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
# Controller / execution layer
#
# The Runner module acts as the orchestration layer that
# combines integration rules, error estimation, and fitting
# logic into a single workflow entry point.
# ============================================================
include("controller/Runner.jl")

using .Runner

end  # module Maranatha