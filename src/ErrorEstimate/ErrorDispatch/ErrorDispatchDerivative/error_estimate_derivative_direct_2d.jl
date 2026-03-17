# ============================================================================
# src/ErrorEstimate/ErrorDispatch/error_estimate_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_2d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    )

Estimate a ``2``-dimensional axis-separable midpoint-residual truncation-error model.

# Function description
This routine applies the ``1``-dimensional midpoint error operator along each
axis of the square domain ``[a,b]^2`` and integrates the resulting derivative
slices over the remaining axis.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref), which reuses previously constructed
residual data for the same rule configuration when available.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} \\right) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x, y)``.
- `a::Real`: Lower bound.
- `b::Real`: Upper bound.
- `N::Int`: Number of subintervals per axis.
- `rule::Symbol`: Quadrature rule symbol.
- `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments
- `err_method::Symbol`: Derivative backend selector.
- `nerr_terms::Int`: Number of nonzero residual terms to include.
- `kmax::Int`: Maximum residual order scanned.

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
- Propagates quadrature-node construction, residual-model extraction, and
  derivative-evaluation errors.

# Notes
- Only axis-separable contributions are modeled.
- Mixed derivative terms are intentionally omitted.
- Residual-term reuse through caching reduces repeated setup cost across
  multiple calls with the same rule configuration.
"""
function error_estimate_2d(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2
    ȳ = (aa + bb) / 2

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    ks, coeffs, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    @inbounds for it in eachindex(ks)
        k = ks[it]

        if k == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]

        I1 = 0.0
        for j in eachindex(xs)
            y = xs[j]
            gx(x) = f(x, y)

            I1 += wx[j] * nth_derivative(
                gx, x̄, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:x, stage=:midpoint,
                err_method=err_method
            )
        end

        I2 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            gy(y) = f(x, y)

            I2 += wx[i] * nth_derivative(
                gy, ȳ, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:y, stage=:midpoint,
                err_method=err_method
            )
        end

        derivatives[it] = I1 + I2
        terms[it] = coeff * h^(k + 1) * derivatives[it]
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = (x̄, ȳ),
        h           = h
    )
end

"""
    error_estimate_2d_jet(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    )

Estimate a ``2``-dimensional axis-separable midpoint-residual truncation-error
model using derivative-jet reuse.

# Function description
This routine builds the same ``2``-dimensional asymptotic midpoint-residual
error model as [`error_estimate_2d`](@ref), but instead of requesting each
derivative order independently, it evaluates the required derivatives through
shared jet-based calls along each axis slice.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`_derivative_values_for_ks`](@ref) to each
slice function in order to reuse one derivative jet per slice rather than one
scalar derivative call per requested order.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} \\right) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x, y)``.
- `a::Real`: Lower bound.
- `b::Real`: Upper bound.
- `N::Int`: Number of subintervals per axis.
- `rule::Symbol`: Quadrature rule symbol.
- `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments
- `err_method::Symbol`: Derivative backend selector.
- `nerr_terms::Int`: Number of nonzero residual terms to include.
- `kmax::Int`: Maximum residual order scanned.

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
- Propagates quadrature-node construction, residual-model extraction, and
  jet-based derivative-evaluation errors.

# Notes
- Only axis-separable contributions are modeled.
- Mixed derivative terms are intentionally omitted.
- This variant is especially useful when several residual orders are needed for
  each slice, since one shared jet can supply all requested derivatives on that
  slice.
"""
function error_estimate_2d_jet(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2
    ȳ = (aa + bb) / 2

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    # ks, coeffs, _center = _leading_residual_terms_any(
    #     rule, boundary, N;
    #     nterms = nerr_terms,
    #     kmax   = kmax
    # )
    ks, coeffs, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = zeros(Float64, n)
    terms       = zeros(Float64, n)

    @inbounds for j in eachindex(xs)
        y = xs[j]
        gx(x) = f(x, y)

        vals = _derivative_values_for_ks(
            gx,
            x̄,
            ks;
            h = h,
            rule = rule,
            N = N,
            dim = 2,
            err_method = err_method,
            side = :mid,
            axis = :x,
            stage = :midpoint,
        )

        for it in eachindex(ks)
            k = ks[it]
            k == 0 && continue
            derivatives[it] += wx[j] * vals[it]
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        gy(y) = f(x, y)

        vals = _derivative_values_for_ks(
            gy,
            ȳ,
            ks;
            h = h,
            rule = rule,
            N = N,
            dim = 2,
            err_method = err_method,
            side = :mid,
            axis = :y,
            stage = :midpoint,
        )

        for it in eachindex(ks)
            k = ks[it]
            k == 0 && continue
            derivatives[it] += wx[i] * vals[it]
        end
    end

    @inbounds for it in eachindex(ks)
        k = ks[it]
        if k == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
        else
            terms[it] = coeffs[it] * h^(k + 1) * derivatives[it]
        end
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = (x̄, ȳ),
        h           = h
    )
end