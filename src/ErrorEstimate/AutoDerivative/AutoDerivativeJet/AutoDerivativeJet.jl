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

@inline function resolve_derivative_jet_backend(err_method::Symbol)
    return err_method === :forwarddiff         ? (ADForwardDiff.derivative_jet_forwarddiff, :forwarddiff) :
           err_method === :taylorseries        ? (ADTaylorSeries.derivative_jet_taylor, :taylorseries) :
           err_method === :fastdifferentiation ? (ADFastDifferentiation.derivative_jet_fastdifferentiation, :fastdifferentiation) :
           err_method === :enzyme              ? (ADEnzyme.derivative_jet_enzyme, :enzyme) :
           JobLoggerTools.error_benji("Unknown err_method=$err_method")
end

"""
    derivative_jet(
        g,
        x,
        nmax;
        h,
        rule,
        N,
        dim::Int,
        err_method::Symbol = :forwarddiff,
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :midpoint
    ) -> Vector{Float64}

Compute or retrieve the derivative jet of `g` at `x` up to order `nmax`
using a selected backend.

# Function description
This function is the jet-level dispatcher used by the error-estimation system.
It first checks [`_DERIV_JET_CACHE`](@ref) for a previously computed jet
with the same callable, evaluation point, maximum order, and backend. If a
cached jet is found, it is returned immediately.

Otherwise, the function dispatches according to `err_method`, stores the newly
computed jet in the cache, and returns it:

- `:forwarddiff`         → [`derivative_jet_forwarddiff`](@ref)
- `:taylorseries`        → [`derivative_jet_taylor`](@ref)
- `:fastdifferentiation` → [`derivative_jet_fastdifferentiation`](@ref)
- `:enzyme`              → [`derivative_jet_enzyme`](@ref)

If an unknown backend symbol is provided, the function aborts through
[`JobLoggerTools.error_benji`](@ref) with detailed context.

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
- `Vector{Float64}`:
  `[g(x), g'(x), g''(x), ..., g^(nmax)(x)]`.

# Notes
- This function is marked `@inline` to reduce dispatch overhead in tight loops.
- The cache key uses the callable object `g` directly, together with `x`,
  `nmax`, and `err_method`.
- This routine is especially useful when several derivative orders at the same
  point are required, since one jet can then serve multiple requests.
"""
@inline function derivative_jet(
    jet_fun,
    backend_tag::Symbol,
    g,
    x,
    nmax;
    h,
    rule,
    N,
    dim::Int,
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)
    x0 = float(x)
    key = (g, x0, nmax, backend_tag)

    if haskey(_DERIV_JET_CACHE, key)
        return _DERIV_JET_CACHE[key]
    end

    jet = jet_fun(g, x0, nmax)

    _DERIV_JET_CACHE[key] = jet
    return jet
end

"""
    _derivative_values_for_ks(
        g,
        x0,
        ks::AbstractVector{<:Integer};
        h,
        rule,
        N,
        dim::Int,
        err_method::Symbol = :forwarddiff,
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :midpoint
    ) -> Vector{Float64}

Return selected derivative values of `g` at `x0` for the derivative orders
listed in `ks`.

# Function description
This helper computes a derivative jet of `g` up to the maximum order appearing
in `ks`, then extracts only the requested derivative orders and returns them as
a dense `Float64` vector.

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
- `g`: Scalar callable.
- `x0`: Evaluation point.
- `ks::AbstractVector{<:Integer}`: Requested derivative orders.

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
- `Vector{Float64}`:
  A vector containing the requested derivative values in the same order as `ks`.

# Notes
- If `ks` is empty, the function returns `Float64[]`.
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
    h,
    rule,
    N,
    dim::Int,
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)
    isempty(ks) && return Float64[]

    nmax = maximum(ks)

    jet = derivative_jet(
        jet_fun,
        backend_tag,
        g,
        x0,
        nmax;
        h = h,
        rule = rule,
        N = N,
        dim = dim,
        side = side,
        axis = axis,
        stage = stage,
    )

    vals = Vector{Float64}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        k = ks[i]
        vals[i] = float(jet[k + 1])
    end

    return vals
end

end  # module AutoDerivativeJet