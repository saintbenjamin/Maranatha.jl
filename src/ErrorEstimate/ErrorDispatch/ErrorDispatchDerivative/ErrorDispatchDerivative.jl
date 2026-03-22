# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/ErrorDispatchDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorDispatchDerivative

Unified derivative-based error-estimation dispatch layer.

# Module description
`ErrorDispatchDerivative` provides the shared orchestration for residual-based
error estimators that combine quadrature residual models with automatic
derivative probes.

Its responsibilities include:

- selecting direct versus jet derivative workflows,
- validating scalar versus axis-wise `rule` / `boundary` specifications,
- requesting family-specific residual models,
- combining per-axis residual contributions into unified result objects.

This module is the main bridge between the automatic-differentiation layer and
the rule-family residual backends.

# Notes
- This is an internal module.
- The generic `*_nd` paths preserve per-axis results; some lower-dimensional
  wrappers additionally expose legacy flattened fields for compatibility.
"""
module ErrorDispatchDerivative

import ..JobLoggerTools
import ..QuadratureBoundarySpec
import ..QuadratureRuleSpec
import ..NewtonCotes
import ..Gauss
import ..BSpline
import ..QuadratureNodes
import ..AutoDerivativeDirect
import ..AutoDerivativeJet
import ..ErrorNewtonCotesDerivative
import ..ErrorGaussDerivative
import ..ErrorBSplineDerivative
import .._RES_MODEL_CACHE
import .._NTH_DERIV_CACHE
import .._DERIV_JET_CACHE

include("internal/clear_error_estimate_derivative_caches!.jl")
include("internal/_get_residual_model_fixed.jl")
include("internal/_leading_residual_terms_any.jl")
include("internal/_leading_residual_ks_with_center_any.jl")
include("internal/_flatten_axiswise_error_result.jl")
include("internal/_resolve_error_estimate_derivative_type.jl")
include("internal/_dispatch_error_estimate_derivative_direct_by_dim.jl")
include("internal/_dispatch_error_estimate_derivative_jet_by_dim.jl")

include("error_estimate_derivative_direct_1d.jl")
include("error_estimate_derivative_direct_2d.jl")
include("error_estimate_derivative_direct_3d.jl")
include("error_estimate_derivative_direct_4d.jl")
include("error_estimate_derivative_direct_nd.jl")
include("error_estimate_derivative_direct.jl")

include("error_estimate_derivative_jet_1d.jl")
include("error_estimate_derivative_jet_2d.jl")
include("error_estimate_derivative_jet_3d.jl")
include("error_estimate_derivative_jet_4d.jl")
include("error_estimate_derivative_jet_nd.jl")
include("error_estimate_derivative_jet.jl")

end  # module ErrorDispatchDerivative
