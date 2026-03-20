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
        a,
        b,
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
error model as [`error_estimate_derivative_direct_3d`](@ref), but instead of
requesting each derivative order independently, it evaluates the required
derivatives through shared jet-based calls along each axis slice.

Two domain conventions are supported:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as ``[a,b]^3``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `3`, the domain is interpreted as
  ``[a_1,b_1] \\times [a_2,b_2] \\times [a_3,b_3]``.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`AutoDerivativeJet._derivative_values_for_ks`](@ref)
to each slice function in order to reuse one derivative jet per slice rather
than one scalar derivative call per requested order.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} \\right) \\, .
```

# Arguments

* `f`: Scalar callable integrand `f(x, y, z)`.
* `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a length-3
  tuple/vector of per-axis lower bounds.
* `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a length-3
  tuple/vector of per-axis upper bounds.
* `N::Int`: Number of subintervals per axis.
* `rule::Symbol`: Quadrature rule symbol.
* `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments

* `err_method::Symbol`: Derivative backend selector.
* `nerr_terms::Int`: Number of nonzero residual terms to include.
* `kmax::Int`: Maximum residual order scanned.
* `real_type = nothing`:
  Optional scalar type used internally for bound conversion, quadrature nodes
  and weights, residual-coefficient conversion, and jet-based derivative
  evaluation.

# Returns

* `NamedTuple` with fields:

  * `ks`
  * `coeffs`
  * `derivatives`
  * `terms`
  * `total`
  * `center`
  * `h`

# Errors

* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 3`
  or `length(b) != 3`.
* Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or `kmax < 0`.
* Propagates quadrature-node construction, residual-model extraction, and
  jet-based derivative-evaluation errors.

# Notes

* Only axis-separable contributions are included.
* Mixed derivative terms are intentionally omitted.
* This variant is especially useful when several residual orders are needed for
  each slice, since one shared jet can supply all requested derivatives on that
  slice.
"""
function error_estimate_derivative_jet_3d(
    f,
    a,
    b,
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

    # Domain handling (hypercube or per-axis)
    if !(a isa AbstractVector || a isa Tuple)
        ax = ay = az = convert(T, a)
        bx = by = bz = convert(T, b)
    else
        length(a) == 3 || throw(ArgumentError("length(a) must be 3"))
        length(b) == 3 || throw(ArgumentError("length(b) must be 3"))
        ax, ay, az = convert(T, a[1]), convert(T, a[2]), convert(T, a[3])
        bx, by, bz = convert(T, b[1]), convert(T, b[2]), convert(T, b[3])
    end

    hx = (bx - ax) / T(N)
    hy = (by - ay) / T(N)
    hz = (bz - az) / T(N)

    x̄ = (ax + bx) / T(2)
    ȳ = (ay + by) / T(2)
    z̄ = (az + bz) / T(2)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        ax, 
        bx, 
        N, 
        rule, 
        boundary; 
        real_type = T
    )
    ys, wy = QuadratureNodes.get_quadrature_1d_nodes_weights(
        ay, 
        by, 
        N, 
        rule, 
        boundary; 
        real_type = T
    )
    zs, wz = QuadratureNodes.get_quadrature_1d_nodes_weights(
        az, 
        bz, 
        N, 
        rule, 
        boundary; 
        real_type = T
    )

    ks, coeffs0, _ = _get_residual_model_fixed(
        rule, 
        boundary, 
        N; 
        nterms = nerr_terms, 
        kmax = kmax
    )
    coeffs = T.(coeffs0)

    derivatives = zeros(T, length(ks))
    terms       = zeros(T, length(ks))

    jet_fun, backend_tag = AutoDerivativeJet.resolve_derivative_jet_backend(err_method)

    # X-axis derivatives integrated over Y,Z
    @inbounds for j in eachindex(ys)
        y = ys[j]; wyj = wy[j]
        for k2 in eachindex(zs)
            z = zs[k2]; w = wyj * wz[k2]
            gx(x) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun, 
                backend_tag, 
                gx, 
                x̄, 
                ks;
            )
            vals = T.(vals0)

            for it in eachindex(ks)
                kk = ks[it]; kk == 0 && continue
                derivatives[it] += w * vals[it]
            end
        end
    end

    # Y-axis derivatives integrated over X,Z
    @inbounds for i in eachindex(xs)
        x = xs[i]; wxi = wx[i]
        for k2 in eachindex(zs)
            z = zs[k2]; w = wxi * wz[k2]
            gy(y) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun, 
                backend_tag, 
                gy, 
                ȳ, 
                ks;
            )
            vals = T.(vals0)

            for it in eachindex(ks)
                kk = ks[it]; kk == 0 && continue
                derivatives[it] += w * vals[it]
            end
        end
    end

    # Z-axis derivatives integrated over X,Y
    @inbounds for i in eachindex(xs)
        x = xs[i]; wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]; w = wxi * wy[j]
            gz(z) = f(x, y, z)

            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun, 
                backend_tag, 
                gz, 
                z̄, 
                ks;
            )
            vals = T.(vals0)

            for it in eachindex(ks)
                kk = ks[it]; kk == 0 && continue
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
            terms[it] = coeffs[it] * (hx + hy + hz)^(kk + 1) * derivatives[it]
        end
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = (x̄, ȳ, z̄),
        h           = (hx, hy, hz)
    )
end