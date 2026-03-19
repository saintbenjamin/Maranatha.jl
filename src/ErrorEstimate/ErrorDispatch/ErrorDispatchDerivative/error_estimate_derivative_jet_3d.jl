# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet_3d(
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

Estimate a ``3``-dimensional axis-separable midpoint-residual truncation-error
model using derivative-jet reuse.

# Function description
This routine builds the same ``3``-dimensional asymptotic midpoint-residual
error model as [`error_estimate_derivative_direct_3d`](@ref), but instead of requesting each
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
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion, quadrature nodes
  and weights, residual-coefficient conversion, and jet-based derivative
  evaluation.

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
- Only axis-separable contributions are included.
- Mixed derivative terms are intentionally omitted.
- This variant is especially useful when several residual orders are needed for
  each slice, since one shared jet can supply all requested derivatives on that
  slice.
"""
function error_estimate_derivative_jet_3d(
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
    ȳ = (aa + bb) / T(2)
    z̄ = (aa + bb) / T(2)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        aa, bb, N, rule, boundary;
        real_type = T,
    )
    ys, wy = xs, wx
    zs, wz = xs, wx

    ks, coeffs0, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )
    coeffs = T.(coeffs0)

    n = length(ks)

    derivatives = zeros(T, n)
    terms       = zeros(T, n)

    jet_fun, backend_tag =
        AutoDerivativeJet.resolve_derivative_jet_backend(err_method)

    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k2 in eachindex(zs)
            z = zs[k2]
            gx(x) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun,
                backend_tag,
                gx,
                x̄,
                ks;
            )
            vals = T.(vals0)

            w = wyj * wz[k2]
            for it in eachindex(ks)
                kk = ks[it]
                kk == 0 && continue
                derivatives[it] += w * vals[it]
            end
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k2 in eachindex(zs)
            z = zs[k2]
            gy(y) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun,
                backend_tag,
                gy,
                ȳ,
                ks;

            )
            vals = T.(vals0)

            w = wxi * wz[k2]
            for it in eachindex(ks)
                kk = ks[it]
                kk == 0 && continue
                derivatives[it] += w * vals[it]
            end
        end
    end

    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            gz(z) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun,
                backend_tag,
                gz,
                z̄,
                ks;

            )
            vals = T.(vals0)

            w = wxi * wy[j]
            for it in eachindex(ks)
                kk = ks[it]
                kk == 0 && continue
                derivatives[it] += w * vals[it]
            end
        end
    end

    @inbounds for it in eachindex(ks)
        kk = ks[it]
        if kk == 0
            derivatives[it] = zero(T)
            terms[it] = zero(T)
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
        center      = (x̄, ȳ, z̄),
        h           = h
    )
end