# ============================================================================
# src/Documentation/Documentation.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Documentation

Documentation and reporting utilities for `Maranatha.jl`.

`Maranatha.Documentation` collects the package's higher-level output helpers
for visualization and report generation.

The main submodules are:

- [`PlotTools`](@ref)
- [`Reporter`](@ref)

`PlotTools` provides plotting utilities for convergence results, raw
datapoints, quadrature coverage, and PyPlot styling.

`Reporter` provides LaTeX/Markdown summary generators, internal-note
builders, and related reporting helpers for convergence studies.
"""
module Documentation

import ..PyPlot
import ..LinearAlgebra
import ..Printf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter
import ..Utils.MaranathaTOML
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..Quadrature.QuadratureDispatch

include("PlotTools.jl")
include("Reporter.jl")

using .PlotTools
using .Reporter

end  # module Documentation