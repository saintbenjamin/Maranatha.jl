# ============================================================================
# src/ErrorEstimate/ErrorEstimate.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorEstimate

using ..LinearAlgebra
using ..TaylorSeries
using ..Enzyme
using ..ForwardDiff

using ..Utils.JobLoggerTools
using ..Quadrature

include("ErrorNewtonCotes.jl")
include("ErrorGauss.jl")
include("ErrorBSpline.jl")
include("ErrorDispatch.jl")

using .ErrorNewtonCotes
using .ErrorGauss
using .ErrorBSpline
using .ErrorDispatch

export error_estimate, error_estimate_threads

end  # module ErrorEstimate