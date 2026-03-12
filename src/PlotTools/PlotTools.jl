# ============================================================================
# src/PlotTools/PlotTools.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module PlotTools

Plotting utilities for `Maranatha.jl`.

`Maranatha.PlotTools` provides visualization helpers for fitted convergence
results, raw convergence datapoints, pedagogical 1D quadrature coverage, and
global PyPlot styling.

The main public entry points are:

- [`plot_convergence_result`](@ref)
- [`plot_datapoints_result`](@ref)
- [`plot_quadrature_coverage_1d`](@ref)

This module also provides internal helpers such as
[`_smart_text_placement!`](@ref).
"""
module PlotTools

import ..PyPlot
import ..LinearAlgebra
import ..Printf: @sprintf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter
import ..Utils.MaranathaTOML
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..Quadrature.QuadratureDispatch

include("set_pyplot_latex_style.jl")
include("_smart_text_placement.jl")
include("plot_convergence_result.jl")
include("plot_datapoints_result.jl")
include("plot_quadrature_coverage_1d.jl")

end  # module PlotTools