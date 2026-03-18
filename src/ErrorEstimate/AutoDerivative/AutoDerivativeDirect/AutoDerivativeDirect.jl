# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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

@inline function resolve_nth_derivative_backend(err_method::Symbol)
    return err_method === :forwarddiff         ? (ADForwardDiff.nth_derivative_forwarddiff, :forwarddiff) :
           err_method === :taylorseries        ? (ADTaylorSeries.nth_derivative_taylor, :taylorseries) :
           err_method === :fastdifferentiation ? (ADFastDifferentiation.nth_derivative_fastdifferentiation, :fastdifferentiation) :
           err_method === :enzyme              ? (ADEnzyme.nth_derivative_enzyme, :enzyme) :
           JobLoggerTools.error_benji("Unknown err_method=$err_method")
end

"""
    nth_derivative(
        g,
        x,
        n;
        h,
        rule,
        N,
        dim::Int,
        err_method::Symbol = :forwarddiff,
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :midpoint
    ) -> Real

Compute the `n`-th derivative of a scalar callable `g` at point `x`
using a selected differentiation backend.

# Function description
This function is a lightweight backend dispatcher used by the error-estimation
pipeline. It first checks the global scalar-derivative cache and, if a cached
value is available for the same callable, evaluation point, derivative order,
and backend, returns that cached result immediately.

If no cached value is found, the function dispatches according to `err_method`
and stores the computed result in [`_NTH_DERIV_CACHE`](@ref):

- `:forwarddiff`         → [`nth_derivative_forwarddiff`](@ref)
- `:taylorseries`        → [`nth_derivative_taylor`](@ref)
- `:fastdifferentiation` → [`nth_derivative_fastdifferentiation`](@ref)
- `:enzyme`              → [`nth_derivative_enzyme`](@ref)

If an unknown `err_method` is provided, the function aborts via
[`JobLoggerTools.error_benji`](@ref) with a context-rich message.

This interface is shared across ``1``/``2``/``3``/``4``-dimensional and general
``n``-dimensional error estimators.

# Keyword arguments
- `h`          : Grid spacing.
- `rule`       : Quadrature rule symbol.
- `N`          : Number of subdivisions.
- `dim::Int`   : Problem dimensionality.
- `err_method` : Backend selector
  (`:forwarddiff | :taylorseries | :fastdifferentiation | :enzyme`).
- `side`       : Boundary-location indicator (`:L`, `:R`, or `:mid`).
- `axis`       : Axis index or symbolic name.
- `stage`      : Stage tag for logging (e.g. `:midpoint` or `:boundary`).

# Returns
- The `n`-th derivative value ``g^{(n)}(x)`` as returned by the selected
  backend or from cache.

# Notes
- This function is marked `@inline` so it can be inlined into tight quadrature
  loops with minimal dispatch overhead.
- The cache key uses `objectid(g)`, the floating-point evaluation point, the
  derivative order, and the backend symbol.
- Any finiteness checks or higher-level fallback policies must be implemented
  outside this dispatcher.
"""
@inline function nth_derivative(
    deriv_fun,
    backend_tag::Symbol,  
    g,
    x,
    n;
    h,
    rule,
    N,
    dim::Int,
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)
    x0 = float(x)
    key = (objectid(g), x0, n, backend_tag)

    if haskey(_NTH_DERIV_CACHE, key)
        return _NTH_DERIV_CACHE[key]
    end

    val = deriv_fun(g, x0, n)
    _NTH_DERIV_CACHE[key] = val
    return val
end

end  # module AutoDerivativeDirect