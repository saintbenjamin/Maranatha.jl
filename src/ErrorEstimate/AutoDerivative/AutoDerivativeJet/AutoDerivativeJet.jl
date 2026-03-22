# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeJet/AutoDerivativeJet.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module AutoDerivativeJet

Derivative-jet dispatch layer for the error-estimation subsystem.

# Module description
`AutoDerivativeJet` unifies the backend-specific routines that compute the full
derivative jet
`[f(x), f'(x), ..., f^(nmax)(x)]`
at a single scalar point.

Its responsibilities include:

- selecting a concrete jet backend from `err_method`,
- exposing a uniform derivative-jet interface,
- coordinating cache-aware jet evaluation for derivative-based residual
  estimators.

# Notes
- This is an internal module.
- Supported backends are implemented in the sibling submodules included here.
"""
module AutoDerivativeJet

import ..JobLoggerTools
import .._RES_MODEL_CACHE
import .._NTH_DERIV_CACHE
import .._DERIV_JET_CACHE

include("ADTaylorSeries.jl")
include("ADEnzyme.jl")
include("ADForwardDiff.jl")
include("ADFastDifferentiation.jl")

using .ADTaylorSeries
using .ADEnzyme
using .ADForwardDiff
using .ADFastDifferentiation

include("internal/_resolve_derivative_jet_backend.jl")
include("internal/_derivative_jet.jl")
include("internal/_derivative_values_for_ks.jl")

end  # module AutoDerivativeJet
