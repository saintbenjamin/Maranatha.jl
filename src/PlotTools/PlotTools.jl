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

using ..PyPlot
using ..LinearAlgebra
using ..Printf

using ..Utils.JobLoggerTools
using ..Utils.AvgErrFormatter
using ..Quadrature
using ..ErrorEstimate

export set_pyplot_latex_style, plot_convergence_result, plot_quadrature_coverage_1d

include("set_pyplot_latex_style.jl")
include("_smart_text_placement.jl")
include("plot_convergence_result.jl")
include("plot_quadrature_coverage_1d.jl")

end  # module PlotTools