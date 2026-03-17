# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet_4d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet_4d(
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

Estimate a ``4``-dimensional axis-separable midpoint-residual truncation-error
model using derivative-jet reuse.

# Function description
This routine builds the same ``4``-dimensional asymptotic midpoint-residual
error model as [`error_estimate_derivative_direct_4d`](@ref), but instead of requesting each
derivative order independently, it evaluates the required derivatives through
shared jet-based calls along each axis slice.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`AutoDerivativeJet._derivative_values_for_ks`](@ref) to each
slice function in order to reuse one derivative jet per slice rather than one
scalar derivative call per requested order.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} + I_t^{(k_i)} \\right) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x, y, z, t)``.
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
function error_estimate_derivative_jet_4d(
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
    t̄ = (aa + bb) / 2

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

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

    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k2 in eachindex(zs)
            z = zs[k2]
            wyj_wzk = wyj * wz[k2]
            for l in eachindex(ts)
                t = ts[l]
                gx(x) = f(x, y, z, t)

                vals = AutoDerivativeJet._derivative_values_for_ks(
                    gx,
                    x̄,
                    ks;
                    h = h,
                    rule = rule,
                    N = N,
                    dim = 4,
                    err_method = err_method,
                    side = :mid,
                    axis = :x,
                    stage = :midpoint,
                )

                w = wyj_wzk * wt[l]
                for it in eachindex(ks)
                    kk = ks[it]
                    kk == 0 && continue
                    derivatives[it] += w * vals[it]
                end
            end
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k2 in eachindex(zs)
            z = zs[k2]
            wxi_wzk = wxi * wz[k2]
            for l in eachindex(ts)
                t = ts[l]
                gy(y) = f(x, y, z, t)

                vals = AutoDerivativeJet._derivative_values_for_ks(
                    gy,
                    ȳ,
                    ks;
                    h = h,
                    rule = rule,
                    N = N,
                    dim = 4,
                    err_method = err_method,
                    side = :mid,
                    axis = :y,
                    stage = :midpoint,
                )

                w = wxi_wzk * wt[l]
                for it in eachindex(ks)
                    kk = ks[it]
                    kk == 0 && continue
                    derivatives[it] += w * vals[it]
                end
            end
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for l in eachindex(ts)
                t = ts[l]
                gz(z) = f(x, y, z, t)

                vals = AutoDerivativeJet._derivative_values_for_ks(
                    gz,
                    z̄,
                    ks;
                    h = h,
                    rule = rule,
                    N = N,
                    dim = 4,
                    err_method = err_method,
                    side = :mid,
                    axis = :z,
                    stage = :midpoint,
                )

                w = wxi_wyj * wt[l]
                for it in eachindex(ks)
                    kk = ks[it]
                    kk == 0 && continue
                    derivatives[it] += w * vals[it]
                end
            end
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for k2 in eachindex(zs)
                z = zs[k2]
                gt(t) = f(x, y, z, t)

                vals = AutoDerivativeJet._derivative_values_for_ks(
                    gt,
                    t̄,
                    ks;
                    h = h,
                    rule = rule,
                    N = N,
                    dim = 4,
                    err_method = err_method,
                    side = :mid,
                    axis = :t,
                    stage = :midpoint,
                )

                w = wxi_wyj * wz[k2]
                for it in eachindex(ks)
                    kk = ks[it]
                    kk == 0 && continue
                    derivatives[it] += w * vals[it]
                end
            end
        end
    end

    @inbounds for it in eachindex(ks)
        kk = ks[it]
        if kk == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
        else
            terms[it] = coeffs[it] * h^(kk + 1) * derivatives[it]
        end
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = (x̄, ȳ, z̄, t̄),
        h           = h
    )
end