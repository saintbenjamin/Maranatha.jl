# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/internal/_nth_derivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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

# Errors
- Propagates backend failures from `deriv_fun`.
- Propagates conversion errors if `x` or the backend result cannot be converted
  to the active scalar type.

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
