# ============================================================================
# src/Quadrature/Quadrature.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Quadrature

using ..LinearAlgebra

using ..Utils.JobLoggerTools

export quadrature, get_quadrature_1d_nodes_weights

include("NewtonCotes.jl")
include("Gauss.jl")
include("BSpline.jl")
include("QuadratureDispatch.jl")

using .NewtonCotes
using .Gauss
using .BSpline
using .QuadratureDispatch

end  # module Quadrature