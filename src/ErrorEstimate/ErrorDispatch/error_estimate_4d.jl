# ============================================================================
# src/ErrorEstimate/ErrorDispatch/error_estimate_4d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_4d(
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

Estimate a `4`-dimensional axis-separable midpoint-residual truncation-error model.

# Function description
This routine applies the `1`-dimensional midpoint error operator along each axis
of the hypercube `[a,b]^4` and integrates the resulting derivative slices over
the remaining three axes.

For each collected residual order `k`, it forms the model contribution
`coeff_k * h^(k+1) * (I_x^(k) + I_y^(k) + I_z^(k) + I_t^(k))`.

# Arguments
- `f`: Scalar callable integrand `f(x, y, z, t)`.
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
- Throws (via `JobLoggerTools.error_benji`) if `nerr_terms < 1` or `kmax < 0`.
- Propagates quadrature-node construction, residual-extraction, and derivative-evaluation errors.

# Notes
- Only axis-separable contributions are modeled.
- Mixed derivative terms are intentionally omitted.
"""
function error_estimate_4d(
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

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

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
                wyj_wzk = wyj * wz[k2]
                for l in eachindex(ts)
                    t = ts[l]
                    gx(x) = f(x, y, z, t)
                    I1 += wyj_wzk * wt[l] * nth_derivative(
                        gx, x̄, kk;
                        h=h, rule=rule, N=N, dim=4,
                        side=:mid, axis=:x, stage=:midpoint,
                        err_method=err_method
                    )
                end
            end
        end

        I2 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k2 in eachindex(zs)
                z = zs[k2]
                wxi_wzk = wxi * wz[k2]
                for l in eachindex(ts)
                    t = ts[l]
                    gy(y) = f(x, y, z, t)
                    I2 += wxi_wzk * wt[l] * nth_derivative(
                        gy, ȳ, kk;
                        h=h, rule=rule, N=N, dim=4,
                        side=:mid, axis=:y, stage=:midpoint,
                        err_method=err_method
                    )
                end
            end
        end

        I3 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                wxi_wyj = wxi * wy[j]
                for l in eachindex(ts)
                    t = ts[l]
                    gz(z) = f(x, y, z, t)
                    I3 += wxi_wyj * wt[l] * nth_derivative(
                        gz, z̄, kk;
                        h=h, rule=rule, N=N, dim=4,
                        side=:mid, axis=:z, stage=:midpoint,
                        err_method=err_method
                    )
                end
            end
        end

        I4 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                wxi_wyj = wxi * wy[j]
                for k2 in eachindex(zs)
                    z = zs[k2]
                    gt(t) = f(x, y, z, t)
                    I4 += wxi_wyj * wz[k2] * nth_derivative(
                        gt, t̄, kk;
                        h=h, rule=rule, N=N, dim=4,
                        side=:mid, axis=:t, stage=:midpoint,
                        err_method=err_method
                    )
                end
            end
        end

        derivatives[it] = I1 + I2 + I3 + I4
        terms[it] = coeff * h^(kk + 1) * derivatives[it]
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

"""
    error_estimate_4d_threads(
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

Threaded variant of `error_estimate_4d`.

# Function description
This routine preserves the same 4D midpoint-residual model as
`error_estimate_4d` but parallelizes the dominant flattened 3D index-grid loops
used in the axis-wise cross integrals.

Thread-local partial sums are reduced after each axis contribution is computed.

# Arguments
- Same as `error_estimate_4d`.

# Keyword arguments
- Same as `error_estimate_4d`.

# Returns
- Same `NamedTuple` structure as `error_estimate_4d`.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `nerr_terms < 1` or `kmax < 0`.
- Propagates quadrature-node construction, residual-extraction, and derivative-evaluation errors.

# Notes
- Threading overhead may dominate when the tensor grid is small.
"""
function error_estimate_4d_threads(
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
    ȳ = x̄
    z̄ = x̄
    t̄ = x̄

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)
    L = length(xs)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    @inbounds for it in eachindex(ks)
        kk = ks[it]

        if kk == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]

        I1_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^3)
            tid = Threads.threadid()

            tmp = idx - 1
            j   = (tmp % L) + 1
            tmp ÷= L
            k2  = (tmp % L) + 1
            l   = (tmp ÷ L) + 1

            y = ys[j]
            z = zs[k2]
            t = ts[l]
            w = wy[j] * wz[k2] * wt[l]

            gx = let y=y, z=z, t=t
                x -> f(x, y, z, t)
            end

            dx = nth_derivative(
                gx, x̄, kk;
                h=h, rule=rule, N=N, dim=4,
                side=:mid, axis=:x, stage=:midpoint,
                err_method=err_method
            )

            I1_parts[tid] += w * dx
        end
        I1 = sum(I1_parts)

        I2_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^3)
            tid = Threads.threadid()

            tmp = idx - 1
            i   = (tmp % L) + 1
            tmp ÷= L
            k2  = (tmp % L) + 1
            l   = (tmp ÷ L) + 1

            x = xs[i]
            z = zs[k2]
            t = ts[l]
            w = wx[i] * wz[k2] * wt[l]

            gy = let x=x, z=z, t=t
                y -> f(x, y, z, t)
            end

            dy = nth_derivative(
                gy, ȳ, kk;
                h=h, rule=rule, N=N, dim=4,
                side=:mid, axis=:y, stage=:midpoint,
                err_method=err_method
            )

            I2_parts[tid] += w * dy
        end
        I2 = sum(I2_parts)

        I3_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^3)
            tid = Threads.threadid()

            tmp = idx - 1
            i   = (tmp % L) + 1
            tmp ÷= L
            j   = (tmp % L) + 1
            l   = (tmp ÷ L) + 1

            x = xs[i]
            y = ys[j]
            t = ts[l]
            w = wx[i] * wy[j] * wt[l]

            gz = let x=x, y=y, t=t
                z -> f(x, y, z, t)
            end

            dz = nth_derivative(
                gz, z̄, kk;
                h=h, rule=rule, N=N, dim=4,
                side=:mid, axis=:z, stage=:midpoint,
                err_method=err_method
            )

            I3_parts[tid] += w * dz
        end
        I3 = sum(I3_parts)

        I4_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^3)
            tid = Threads.threadid()

            tmp = idx - 1
            i   = (tmp % L) + 1
            tmp ÷= L
            j   = (tmp % L) + 1
            k2  = (tmp ÷ L) + 1

            x = xs[i]
            y = ys[j]
            z = zs[k2]
            w = wx[i] * wy[j] * wz[k2]

            gt = let x=x, y=y, z=z
                t -> f(x, y, z, t)
            end

            dt = nth_derivative(
                gt, t̄, kk;
                h=h, rule=rule, N=N, dim=4,
                side=:mid, axis=:t, stage=:midpoint,
                err_method=err_method
            )

            I4_parts[tid] += w * dt
        end
        I4 = sum(I4_parts)

        derivatives[it] = I1 + I2 + I3 + I4
        terms[it] = coeff * h^(kk + 1) * derivatives[it]
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