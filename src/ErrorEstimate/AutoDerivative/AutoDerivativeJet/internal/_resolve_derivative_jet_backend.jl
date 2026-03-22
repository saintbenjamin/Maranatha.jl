# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeJet/internal/_resolve_derivative_jet_backend.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    resolve_derivative_jet_backend(
        err_method::Symbol
    ) -> Tuple{Function, Symbol}

Resolve a derivative-jet backend selector into the concrete jet routine and its
canonical backend tag.

# Function description
This helper maps the public derivative-jet backend selector symbol to the
concrete jet-construction routine used by the jet derivative path, together
with the canonical backend tag stored in cache keys and downstream metadata.

# Arguments
- `err_method::Symbol`:
  Backend selector symbol.

  Supported values are:

  - `:forwarddiff`
  - `:taylorseries`
  - `:fastdifferentiation`
  - `:enzyme`

# Returns
- `Tuple{Function, Symbol}`:
  A pair `(jet_fun, backend_tag)` where:

  - `jet_fun` is the backend-specific derivative-jet routine, and
  - `backend_tag` is the normalized backend symbol used for downstream cache keys.

# Errors
- Throws through [`JobLoggerTools.error_benji`](@ref) if `err_method` is not one
  of the supported backend selectors.

# Notes
- This is a lightweight internal dispatcher used by the derivative-jet branch of
  the error-estimation pipeline.
- The returned `backend_tag` is intended to stay consistent with the cache layout
  used by [`derivative_jet`](@ref).
"""
@inline function resolve_derivative_jet_backend(
    err_method::Symbol
)
    return err_method === :forwarddiff         ? (ADForwardDiff.derivative_jet_forwarddiff, :forwarddiff) :
           err_method === :taylorseries        ? (ADTaylorSeries.derivative_jet_taylor, :taylorseries) :
           err_method === :fastdifferentiation ? (ADFastDifferentiation.derivative_jet_fastdifferentiation, :fastdifferentiation) :
           err_method === :enzyme              ? (ADEnzyme.derivative_jet_enzyme, :enzyme) :
           JobLoggerTools.error_benji("Unknown err_method=$err_method")
end
