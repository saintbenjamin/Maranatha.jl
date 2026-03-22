# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_get_residual_model_fixed.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _get_residual_model_fixed(
        rule::Symbol,
        boundary::Symbol,
        Nref::Int;
        nterms::Int,
        kmax::Int,
        real_type = Float64,
    ) -> Tuple{Vector{Int}, Vector, Symbol}

Return a cached residual model for a fixed quadrature configuration.

# Function description
This helper retrieves the leading residual-term model associated with a given
quadrature rule and boundary pattern. If a matching model is already present in
[`_RES_MODEL_CACHE`](@ref), it is returned immediately. Otherwise, the
model is constructed via [`_leading_residual_terms_any`](@ref), stored in the
cache, and then returned.

The returned tuple contains:

- `ks`: indices of the leading nonzero residual terms,
- `coeffs`: corresponding residual coefficients,
- `center`: centering convention tag.

# Arguments
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary-condition symbol.
- `Nref`: Reference subdivision count passed to the residual-term builder.

# Keyword arguments
- `nterms`: Number of leading nonzero residual terms to collect.
- `kmax`: Maximum moment order scanned while searching for residual terms.
- `real_type = Float64`:
  Scalar type used for residual-coefficient conversion and cache separation.

# Returns
- `Tuple{Vector{Int}, Vector, Symbol}`:
  `(ks, coeffs, center)` for the requested residual model, with `coeffs`
  stored in the active scalar type.

# Errors
- Propagates residual-model construction errors from
  [`_leading_residual_terms_any`](@ref).

# Notes
- The cache key is `(rule, boundary, nterms, kmax, real_type)`.
- `Nref` is forwarded to the builder when the model is first created, but it is
  not part of the cache key in the current implementation.
"""
function _get_residual_model_fixed(
    rule::Symbol,
    boundary::Symbol,
    Nref::Int;
    nterms::Int,
    kmax::Int,
    real_type = Float64,
)
    T = real_type
    key = (rule, boundary, nterms, kmax, T)

    if haskey(_RES_MODEL_CACHE, key)
        return _RES_MODEL_CACHE[key]
    end

    ks, coeffs, center = _leading_residual_terms_any(
        rule, boundary, Nref;
        nterms = nterms,
        kmax   = kmax,
        real_type = T,
    )

    _RES_MODEL_CACHE[key] = (ks, coeffs, center)
    return ks, coeffs, center
end
