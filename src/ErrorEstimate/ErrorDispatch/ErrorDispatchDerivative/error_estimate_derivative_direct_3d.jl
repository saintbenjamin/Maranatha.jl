# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_direct_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_direct_3d(
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

Estimate a ``3``-dimensional axis-separable midpoint-residual truncation-error model.

# Function description
This routine applies the ``1``-dimensional midpoint error operator along each
axis of the cube ``[a,b]^3`` and integrates the resulting derivative slices over
the remaining two axes.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref), which reuses previously constructed
residual data for the same rule configuration when available.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} \\right) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x, y, z)``.
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
- Only axis-separable contributions are included.
- Mixed derivative terms are intentionally omitted.
- Residual-term reuse through caching reduces repeated setup cost across
  multiple calls with the same rule configuration.
"""
function error_estimate_derivative_direct_3d(
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
    z̄ = (aa + bb) / 2

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

    ks, coeffs, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    deriv_fun, backend_tag = AutoDerivativeDirect.resolve_nth_derivative_backend(err_method)

    @inbounds for it in eachindex(ks)
        kk = ks[it]

        if kk == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]

        I1 = 0.0
        for j in eachindex(ys)
            y = ys[j]
            wyj = wy[j]
            for k2 in eachindex(zs)
                z = zs[k2]
                gx(x) = f(x, y, z)

                I1 += wyj * wz[k2] * AutoDerivativeDirect.nth_derivative(
                    deriv_fun,
                    backend_tag,
                    gx, x̄, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:x, stage=:midpoint,
                )
            end
        end

        I2 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k2 in eachindex(zs)
                z = zs[k2]
                gy(y) = f(x, y, z)

                I2 += wxi * wz[k2] * AutoDerivativeDirect.nth_derivative(
                    deriv_fun,
                    backend_tag,
                    gy, ȳ, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:y, stage=:midpoint,
                )
            end
        end

        I3 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                gz(z) = f(x, y, z)

                I3 += wxi * wy[j] * AutoDerivativeDirect.nth_derivative(
                    deriv_fun,
                    backend_tag,
                    gz, z̄, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:z, stage=:midpoint,
                )
            end
        end

        derivatives[it] = I1 + I2 + I3
        terms[it] = coeff * h^(kk + 1) * derivatives[it]
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = (x̄, ȳ, z̄),
        h           = h
    )
end