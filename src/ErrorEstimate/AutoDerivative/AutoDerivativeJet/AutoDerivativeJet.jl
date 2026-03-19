# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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

"""
    resolve_derivative_jet_backend(
        err_method::Symbol
    ) -> Tuple{Function, Symbol}

Resolve a derivative-jet backend selector into the concrete jet routine and its
canonical backend tag.

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

"""
    derivative_jet(
        jet_fun,
        backend_tag::Symbol,
        g,
        x,
        nmax;
        real_type = nothing,
    ) -> AbstractVector{<:Real}

Compute or retrieve the derivative jet of `g` at `x` up to order `nmax`
using a pre-resolved backend.

# Function description
This function is the jet-level dispatcher used by the error-estimation system.
It first checks [`_DERIV_JET_CACHE`](@ref) for a previously computed jet
with the same callable, evaluation point, maximum order, backend tag, and
numeric type. If a cached jet is found, it is returned immediately.

Otherwise, the function calls the already resolved jet backend `jet_fun`,
converts the resulting jet to the active scalar type, stores it in the cache,
and returns it.

# Arguments
- `jet_fun`:
  Backend-specific derivative-jet routine, typically returned by
  [`resolve_derivative_jet_backend`](@ref).
- `backend_tag::Symbol`:
  Canonical backend tag used in the jet cache key.
- `g`:
  Scalar callable whose derivative jet is evaluated.
- `x`:
  Evaluation point.
- `nmax`:
  Maximum derivative order included in the returned jet.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used for cache normalization and output conversion.
  If `nothing`, the function uses `typeof(float(x))`.

# Returns
- `AbstractVector{<:Real}`:
  A dense derivative jet
  ``[g(x), g'(x), g''(x), \\ldots, g^{(nmax)}(x)]``
  converted to the active scalar type.

# Notes
- This function is marked `@inline` to reduce dispatch overhead in tight loops.
- The cache key uses the callable `g`, the converted evaluation point,
  `nmax`, the backend tag, and the active scalar type.
- Backend resolution is intentionally performed outside this function, so this
  routine does not inspect `err_method` directly.
- This routine is especially useful when several derivative orders at the same
  point are required, since one jet can then serve multiple requests.
"""
@inline function derivative_jet(
    jet_fun,
    backend_tag::Symbol,
    g,
    x,
    nmax;
    real_type = nothing,
)
    T = isnothing(real_type) ? typeof(float(x)) : real_type
    x0 = convert(T, x)
    key = (g, x0, nmax, backend_tag, T)

    if haskey(_DERIV_JET_CACHE, key)
        return _DERIV_JET_CACHE[key]
    end

    jet0 = jet_fun(g, x0, nmax)
    jet = T.(jet0)

    _DERIV_JET_CACHE[key] = jet
    return jet
end

"""
    _derivative_values_for_ks(
        jet_fun,
        backend_tag::Symbol,
        g,
        x0,
        ks::AbstractVector{<:Integer};
        real_type = nothing,
    ) -> AbstractVector{<:Real}

Return selected derivative values of `g` at `x0` for the derivative orders
listed in `ks`.

# Function description
This helper computes a derivative jet of `g` up to the maximum order appearing
in `ks`, then extracts only the requested derivative orders and returns them as
a dense vector in the active scalar type.

More precisely, if

```julia
ks = [k₁, k₂, ..., k_m]
```

then the returned vector is

```julia
[g^(k₁)(x0), g^(k₂)(x0), ..., g^(k_m)(x0)]
```

with each entry obtained from the shared jet produced by
[`derivative_jet`](@ref). This is useful when several specific derivative
orders are needed at the same point, since one jet can serve all of them.

# Arguments
- `jet_fun`:
  Backend-specific derivative-jet routine, typically returned by
  [`resolve_derivative_jet_backend`](@ref).
- `backend_tag::Symbol`:
  Canonical backend tag used in the jet cache key.
- `g`:
  Scalar callable.
- `x0`:
  Evaluation point.
- `ks::AbstractVector{<:Integer}`:
  Requested derivative orders.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used for jet construction and output conversion.
  If `nothing`, the function uses `typeof(float(x0))`.

# Returns
- `AbstractVector{<:Real}`:
  A vector containing the requested derivative values in the same order as `ks`,
  converted to the active scalar type.

# Notes
- If `ks` is empty, the function returns an empty vector of the active scalar type.
- The derivative jet is computed only up to `maximum(ks)`.
- Since extraction is performed from a shared jet, this helper is typically more
  efficient than requesting each derivative separately.
"""
@inline function _derivative_values_for_ks(
    jet_fun,
    backend_tag::Symbol,
    g,
    x0,
    ks::AbstractVector{<:Integer};
    real_type = nothing,
)
    T = isnothing(real_type) ? typeof(float(x0)) : real_type
    isempty(ks) && return T[]

    nmax = maximum(ks)

    jet = derivative_jet(
        jet_fun,
        backend_tag,
        g,
        x0,
        nmax;
        real_type = T,
    )

    vals = Vector{T}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        k = ks[i]
        vals[i] = jet[k + 1]
    end

    return vals
end

end  # module AutoDerivativeJet