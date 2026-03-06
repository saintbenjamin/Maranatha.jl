# ============================================================================
# src/ErrorEstimate/ErrorDispatch/error_estimate_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_3d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> NamedTuple

Estimate a ``3``-dimensional tensor-product truncation-error *model* for a composite Newton-Cotes rule
on the cube ``[a,b]^3`` using the exact midpoint residual expansion.

# Function description
This routine generalizes the ``1``-dimensional midpoint-residual model to a ``3``-dimensional tensor-product setting
by applying the ``1``-dimensional midpoint error operator along each axis and numerically integrating
the resulting derivative over the remaining axes.

Let ``\\displaystyle{h = \\frac{b-a}{N}}`` and ``\\displaystyle{\\bar{x} = \\bar{y} = \\bar{z} = \\frac{a+b}{2}}``.
From the exact rational composite weights (via midpoint residual moments), we obtain a sequence
of nonzero residual orders ``k`` with exact coefficients
``\\displaystyle{\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}}``. This routine collects the first `nerr_terms`
nonzero residual orders ``k_1, k_2, \\ldots`` (up to `kmax`) and returns the summed separable model:
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i}\\, h^{k_i+1}\\, \\left( I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} \\right),
```
where the axis-wise cross integrals are
```math
I_x^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dy\\,dz\\;
\\frac{\\partial^k f}{\\partial x^k}(\\bar{x}, y, z) \\,,
```
```math
I_y^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dz\\,dx\\;
\\frac{\\partial^k f}{\\partial y^k}(x, \\bar{y}, z) \\,,
```
```math
I_z^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dx\\,dy\\;
\\frac{\\partial^k f}{\\partial z^k}(x, y, \\bar{z}) \\,.
```
Each cross-axis integral is evaluated numerically using the same ``1``-dimensional composite nodes/weights
along the remaining axes.

The routine returns the full decomposition of the asymptotic error model,
including individual residual contributions and their summed value.

# Arguments

* `f`:
  Scalar callable integrand ``f(x,y,z)`` (function, closure, or callable struct).
* `a`, `b`:
  Scalar bounds defining the cube domain ``[a,b]^3``.
* `N`:
  Number of subintervals per axis. Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton-Cotes rule symbol (must be `:newton_pK` style).
* `boundary`:
  Boundary pattern (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).

# Keyword arguments

* `err_method`:
  Backend used for derivative evaluation via [`nth_derivative`](@ref).
  Supported values: `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, `:enzyme`.
* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model (`1` = LO only, `2` = LO+NLO, ...).
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `NamedTuple` with fields:

  * `ks` - residual orders used in the model
  * `coeffs` - midpoint residual coefficients
  * `derivatives` - evaluated derivatives ``f^{(k)}(\\bar{x})``
  * `terms` - individual asymptotic error contributions
  * `total` - summed truncation-error model value
  * `center` - midpoint ``\\bar{x}``
  * `h` - step size

# Errors

* Propagates errors from:

  * midpoint residual extraction / composite weight generation,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.Utils.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
  insufficient nonzero residual terms exist up to `kmax`.

# Notes

* This model sums only *axis-separable* contributions (``x``-only, ``y``-only, ``z``-only operators).
* Mixed derivative terms (e.g. ``\\partial_x^r\\partial_y^s f`` and other cross terms) are higher order
  and intentionally omitted.
* Coefficients are derived in exact rational arithmetic and converted to `Float64` only at the final stage.
"""
function error_estimate_3d(
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

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

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
                gx(x) = f(x, y, z)

                I1 += wyj * wz[k2] * nth_derivative(
                    gx, x̄, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:x, stage=:midpoint,
                    err_method=err_method
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

                I2 += wxi * wz[k2] * nth_derivative(
                    gy, ȳ, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:y, stage=:midpoint,
                    err_method=err_method
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

                I3 += wxi * wy[j] * nth_derivative(
                    gz, z̄, kk;
                    h=h, rule=rule, N=N, dim=3,
                    side=:mid, axis=:z, stage=:midpoint,
                    err_method=err_method
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

"""
    error_estimate_3d_threads(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> NamedTuple

Threaded variant of [`error_estimate_3d`](@ref) for 3D midpoint-residual truncation-error modeling.

All non-threading details (mathematical definition, coefficient construction, residual-term
interpretation, and overall intent) are identical to [`error_estimate_3d`](@ref).
See that function for the full formalism and background.

# Threading implementation

This function parallelizes the dominant axis-wise summation loops using Julia's built-in
multithreading:

* For each residual order `k` in `ks`, the ``x``-, ``y``-, and ``z``-axis contributions are computed
  as weighted sums over 2D index grids (size `length(xs)^2`) corresponding to the remaining axes.
* Each axis sum is distributed via [`Base.Threads.@threads`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads) over a flattened grid-index loop
  (`idx in 1:(L^2)`), which is mapped to the corresponding pair of 1D node indices.
* Each axis sum accumulates into a thread-local `Float64` scratch buffer indexed by
  [`Threads.threadid()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.threadid), followed by a `sum` reduction to form ``I_1``, ``I_2``, and ``I_3``.

Threading is enabled when Julia is started with `JULIA_NUM_THREADS > 1`.

# Arguments

Same as [`error_estimate_3d`](@ref).

# Keyword arguments

Same as [`error_estimate_3d`](@ref).

# Returns

Same as [`error_estimate_3d`](@ref).

# Notes

* This is an asymptotic *model* (fit stabilization / scaling diagnostics), not a strict bound.
* For small `length(xs)` or small `nerr_terms`, threading overhead may dominate.
"""
function error_estimate_3d_threads(
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

    xs, wx = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    L = length(xs)

    @inbounds for it in eachindex(ks)
        kk = ks[it]

        if kk == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]

        I1_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^2)
            tid = Threads.threadid()

            tmp = idx - 1
            j   = (tmp % L) + 1
            k2  = (tmp ÷ L) + 1

            y = ys[j]
            z = zs[k2]
            w = wy[j] * wz[k2]

            gx = let y=y, z=z
                x -> f(x, y, z)
            end

            dx = nth_derivative(
                gx, x̄, kk;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:x, stage=:midpoint,
                err_method=err_method
            )

            I1_parts[tid] += w * dx
        end
        I1 = sum(I1_parts)

        I2_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^2)
            tid = Threads.threadid()

            tmp = idx - 1
            i   = (tmp % L) + 1
            k2  = (tmp ÷ L) + 1

            x = xs[i]
            z = zs[k2]
            w = wx[i] * wz[k2]

            gy = let x=x, z=z
                y -> f(x, y, z)
            end

            dy = nth_derivative(
                gy, ȳ, kk;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:y, stage=:midpoint,
                err_method=err_method
            )

            I2_parts[tid] += w * dy
        end
        I2 = sum(I2_parts)

        I3_parts = zeros(Float64, Threads.maxthreadid())

        Threads.@threads for idx in 1:(L^2)
            tid = Threads.threadid()

            tmp = idx - 1
            i   = (tmp % L) + 1
            j   = (tmp ÷ L) + 1

            x = xs[i]
            y = ys[j]
            w = wx[i] * wy[j]

            gz = let x=x, y=y
                z -> f(x, y, z)
            end

            dz = nth_derivative(
                gz, z̄, kk;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:z, stage=:midpoint,
                err_method=err_method
            )

            I3_parts[tid] += w * dz
        end
        I3 = sum(I3_parts)

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