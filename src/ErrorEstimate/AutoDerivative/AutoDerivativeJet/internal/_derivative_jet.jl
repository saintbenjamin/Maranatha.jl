# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeJet/internal/_derivative_jet.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
