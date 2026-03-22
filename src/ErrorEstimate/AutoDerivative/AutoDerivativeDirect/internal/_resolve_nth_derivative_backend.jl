# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/internal/_resolve_nth_derivative_backend.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    resolve_nth_derivative_backend(
        err_method::Symbol
    ) -> Tuple{Function, Symbol}

Resolve a scalar automatic-differentiation backend selector into the concrete
derivative routine and its canonical backend tag.

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
  A pair `(deriv_fun, backend_tag)` where:

  - `deriv_fun` is the backend-specific scalar `n`-th derivative routine, and
  - `backend_tag` is the normalized backend symbol used for downstream cache keys.

# Errors
- Throws through [`JobLoggerTools.error_benji`](@ref) if `err_method` is not one
  of the supported backend selectors.

# Notes
- This is a lightweight internal dispatcher used by the derivative-based
  error-estimation pipeline.
- The returned `backend_tag` is intended to stay consistent with the cache layout
  used by [`nth_derivative`](@ref).
"""
@inline function resolve_nth_derivative_backend(
    err_method::Symbol
)
    return err_method === :forwarddiff         ? (ADForwardDiff.nth_derivative_forwarddiff, :forwarddiff) :
           err_method === :taylorseries        ? (ADTaylorSeries.nth_derivative_taylor, :taylorseries) :
           err_method === :fastdifferentiation ? (ADFastDifferentiation.nth_derivative_fastdifferentiation, :fastdifferentiation) :
           err_method === :enzyme              ? (ADEnzyme.nth_derivative_enzyme, :enzyme) :
           JobLoggerTools.error_benji("Unknown err_method=$err_method")
end
