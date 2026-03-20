# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_direct_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_direct_1d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128,
        real_type = nothing,
    )

Estimate a ``1``-dimensional midpoint-residual truncation-error model.

# Function description
This routine builds the axis-separable asymptotic error model for the
``1``-dimensional case using the leading nonzero midpoint residual terms of the
selected composite rule.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref), which reuses previously constructed
residual data for the same rule configuration when available.

For each collected residual order ``k_i``, it evaluates the ``k_i``-th
derivative of ``f`` at the physical midpoint and forms the contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, f^{(k_i)}(\\bar{x}) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x)``.
- `a::Real`: Lower bound.
- `b::Real`: Upper bound.
- `N::Int`: Number of subintervals.
- `rule::Symbol`: Quadrature rule symbol.
- `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments
- `err_method::Symbol`: Derivative backend selector.
- `nerr_terms::Int`: Number of nonzero residual terms to include.
- `kmax::Int`: Maximum residual order scanned.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion, midpoint placement,
  residual-coefficient conversion, and derivative evaluation.

# Returns
- `NamedTuple` with fields:
  - `ks`
  - `coeffs`
  - `derivatives`
  - `terms`
  - `total`
  - `center`
  - `h`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or `kmax < 0`.
- Propagates residual-model extraction and derivative-evaluation errors.

# Notes
- This is an asymptotic error model, not a rigorous bound.
- A detected `k == 0` term is suppressed in the returned contribution.
- Residual-term reuse through caching reduces repeated setup cost across
  multiple calls with the same rule configuration.
"""
function error_estimate_derivative_direct_1d(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = convert(T, a)
    bb = convert(T, b)
    h  = (bb - aa) / T(N)
    x̄ = (aa + bb) / T(2)

    ks, coeffs0, _center = _get_residual_model_fixed(
        rule, 
        boundary, 
        N;
        nterms = nerr_terms,
        kmax   = kmax
    )
    coeffs = T.(coeffs0)

    n = length(ks)

    derivatives = Vector{T}(undef, n)
    terms       = Vector{T}(undef, n)

    deriv_fun, backend_tag = AutoDerivativeDirect.resolve_nth_derivative_backend(err_method)

    @inbounds for i in eachindex(ks)
        k = ks[i]

        if k == 0
            derivatives[i] = zero(T)
            terms[i] = zero(T)
            continue
        end

        coeff = coeffs[i]

        dx = AutoDerivativeDirect.nth_derivative(
            deriv_fun,
            backend_tag,
            f,
            x̄, 
            k;
        )

        derivatives[i] = convert(T, dx)
        terms[i] = coeff * h^(k + 1) * derivatives[i]
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = x̄,
        h           = h
    )
end