# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/AutoDerivativeDirect.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module AutoDerivativeDirect

Direct scalar-derivative dispatch layer for the error-estimation subsystem.

# Module description
`AutoDerivativeDirect` unifies the backend-specific routines that compute a
single scalar `n`-th derivative at a given point.

Its responsibilities include:

- selecting a concrete differentiation backend from `err_method`,
- exposing a uniform direct-derivative interface,
- coordinating cache-aware derivative evaluation for residual estimators.

This module sits between the derivative-based error-estimation dispatchers and
the backend-specific AD implementations.

# Notes
- This is an internal module.
- Supported backends are implemented in the sibling submodules included here.
"""
module AutoDerivativeDirect

import ..JobLoggerTools
import .._RES_MODEL_CACHE
import .._NTH_DERIV_CACHE
import .._DERIV_JET_CACHE

include("ADTaylorSeries.jl")
include("ADEnzyme.jl")
include("ADForwardDiff.jl")

using .ADTaylorSeries
using .ADEnzyme
using .ADForwardDiff

include("internal/_resolve_nth_derivative_backend.jl")
include("internal/_nth_derivative.jl")

end  # module AutoDerivativeDirect
