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

import ..LinearAlgebra
import ..TaylorSeries
import ..Enzyme
import ..ForwardDiff
# import ..Diffractor
import ..FastDifferentiation
import ..FastDifferentiation: @variables

import ..JobLoggerTools
import .._RES_MODEL_CACHE
import .._NTH_DERIV_CACHE
import .._DERIV_JET_CACHE

"""
    derivative_jet_taylor(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using `TaylorSeries.jl`.

# Function description
This helper constructs a truncated Taylor polynomial around `x` and extracts
successive derivatives from that representation. The returned vector contains
the function value followed by derivatives of increasing order:

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This backend is often effective when the callable is compatible with
`TaylorSeries.Taylor1` arithmetic and when multiple derivative orders are needed
at the same evaluation point.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates any errors raised by `f` under Taylor-series input.

# Notes
- If `nmax == 0`, the function returns `[f(x)]` as a one-element vector.
- The output is always converted to `Float64`.
"""
function derivative_jet_taylor(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)
    nmax == 0 && return [float(f(x0))]

    t = TaylorSeries.Taylor1(Float64, nmax)
    y = f(x0 + t)

    ders = Vector{Float64}(undef, nmax + 1)
    ders[1] = float(TaylorSeries.constant_term(y))

    tmp = y
    for n in 1:nmax
        tmp = TaylorSeries.derivative(tmp)
        ders[n + 1] = float(TaylorSeries.constant_term(tmp))
    end

    return ders
end

"""
    derivative_jet_forwarddiff(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using repeated `ForwardDiff`
applications.

# Function description
This helper builds higher-order derivatives recursively by repeatedly applying
`ForwardDiff.derivative` to the previously constructed callable. The resulting
vector has the form

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This backend is simple and general for scalar real-to-real callables that are
compatible with `ForwardDiff`.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates backend failures from `ForwardDiff` or the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- This implementation evaluates derivatives sequentially, so very high orders
  may become expensive or numerically delicate.
"""
function derivative_jet_forwarddiff(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)
    ders = Vector{Float64}(undef, nmax + 1)
    ders[1] = float(f(x0))

    g = f
    for n in 1:nmax
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
        ders[n + 1] = float(g(x0))
    end

    return ders
end

"""
    derivative_jet_enzyme(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using repeated `Enzyme` gradients.

# Function description
This helper constructs higher-order derivatives recursively by treating each
previous derivative as a new scalar callable and applying
`Enzyme.gradient(Enzyme.Reverse, ...)` repeatedly. The output vector is

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This provides a jet-oriented interface for Enzyme-based differentiation inside
the error-estimation subsystem.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates errors raised by `Enzyme` or by the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- This routine assumes a scalar output and extracts the single gradient entry
  via `only(...)`.
"""
function derivative_jet_enzyme(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)
    ders = Vector{Float64}(undef, nmax + 1)
    ders[1] = float(f(x0))

    g = f
    for n in 1:nmax
        prev = g
        g = t -> only(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
        ders[n + 1] = float(g(x0))
    end

    return ders
end

"""
    derivative_jet_fastdifferentiation(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using `FastDifferentiation.jl`.

# Function description
This helper constructs a symbolic differentiation graph for `f(t)` and then
builds all derivatives from order `0` through `nmax` with respect to the same
symbolic variable. The compiled FastDifferentiation function is then evaluated
at `x`, and the result is returned as a `Float64` vector.

The output vector is

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

# Arguments
- `f`: Scalar callable compatible with FastDifferentiation symbolic variables.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates symbolic-construction or execution errors from
  `FastDifferentiation` or the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- The internal FastDifferentiation cache is cleared before constructing the
  symbolic derivative chain.
- This backend is useful when symbolic derivative reuse is preferable to
  repeated AD calls.
"""
function derivative_jet_fastdifferentiation(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)

    if nmax == 0
        return [float(f(x0))]
    end

    FastDifferentiation.clear_cache()

    @variables t

    base_expr = f(t)

    exprs = Vector{Any}(undef, nmax + 1)
    exprs[1] = base_expr

    for n in 1:nmax
        exprs[n + 1] = FastDifferentiation.derivative(
            base_expr,
            ntuple(_ -> t, n)...
        )
    end

    exe = FastDifferentiation.make_function(exprs, [t])
    vals = exe(x0)

    return Float64.(vec(vals))
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
)
    x0 = float(x)
    key = (g, x0, nmax, err_method)

    if haskey(_DERIV_JET_CACHE, key)
        return _DERIV_JET_CACHE[key]
    end

    jet =
        if err_method === :forwarddiff
            derivative_jet_forwarddiff(g, x0, nmax)

        elseif err_method === :taylorseries
            derivative_jet_taylor(g, x0, nmax)

        elseif err_method === :fastdifferentiation
            derivative_jet_fastdifferentiation(g, x0, nmax)

        elseif err_method === :enzyme
            derivative_jet_enzyme(g, x0, nmax)

        else
            JobLoggerTools.error_benji(
                "Unknown err_method=$err_method " *
                "h=$h x=$x nmax=$nmax rule=$rule N=$N dim=$dim " *
                "side=$side axis=$axis stage=$stage"
            )
        end

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
)
    isempty(ks) && return Float64[]

    nmax = maximum(ks)

    jet = derivative_jet(
        g,
        x0,
        nmax;
        h = h,
        rule = rule,
        N = N,
        dim = dim,
        err_method = err_method,
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