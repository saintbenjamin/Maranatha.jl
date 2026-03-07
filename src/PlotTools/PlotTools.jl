# ============================================================================
# src/PlotTools/PlotTools.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PlotTools

import ..PyPlot
import ..LinearAlgebra
import ..Printf: @sprintf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter
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