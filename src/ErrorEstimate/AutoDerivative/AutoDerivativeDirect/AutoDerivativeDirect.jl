# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect.jl
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
include("ADFastDifferentiation.jl")

using .ADTaylorSeries
using .ADEnzyme
using .ADForwardDiff
using .ADFastDifferentiation

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

"""
    nth_derivative(
        deriv_fun,
        backend_tag::Symbol,
        g,
        x,
        n;
        real_type = nothing,
    ) -> Real

Compute the `n`-th derivative of a scalar callable `g` at point `x`
using a pre-resolved differentiation backend.

# Function description
This function is a lightweight backend dispatcher used by the error-estimation
pipeline. It first checks the global scalar-derivative cache and, if a cached
value is available for the same callable, evaluation point, derivative order,
backend tag, and numeric type, returns that cached result immediately.

If no cached value is found, the function calls the already resolved backend
function `deriv_fun`, converts the result to the active scalar type, and stores
the computed value in [`_NTH_DERIV_CACHE`](@ref).

This interface is shared across ``1``/``2``/``3``/``4``-dimensional and general
``n``-dimensional error estimators.

# Arguments
- `deriv_fun`:
  Backend-specific scalar derivative routine, typically returned by
  [`resolve_nth_derivative_backend`](@ref).
- `backend_tag::Symbol`:
  Canonical backend tag used in the derivative cache key.
- `g`:
  Scalar callable whose derivative is evaluated.
- `x`:
  Evaluation point.
- `n`:
  Derivative order.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used for cache normalization and output conversion.
  If `nothing`, the function uses `typeof(float(x))`.

# Returns
- `Real`:
  The `n`-th derivative value ``g^{(n)}(x)`` converted to the active scalar type.

# Notes
- This function is marked `@inline` so it can be inlined into tight quadrature
  loops with minimal dispatch overhead.
- The cache key uses `objectid(g)`, the converted evaluation point, the
  derivative order, the backend tag, and the active scalar type.
- Backend resolution is intentionally performed outside this function, so this
  routine does not inspect `err_method` directly.
- Any finiteness checks or higher-level fallback policies must be implemented
  outside this dispatcher.
"""
@inline function nth_derivative(
    deriv_fun,
    backend_tag::Symbol,
    g,
    x,
    n;
    real_type = nothing,
)
    T = isnothing(real_type) ? typeof(float(x)) : real_type
    x0 = convert(T, x)
    key = (objectid(g), x0, n, backend_tag, T)

    if haskey(_NTH_DERIV_CACHE, key)
        return _NTH_DERIV_CACHE[key]
    end

    val = convert(T, deriv_fun(g, x0, n))
    _NTH_DERIV_CACHE[key] = val
    return val
end

end  # module AutoDerivativeDirect
