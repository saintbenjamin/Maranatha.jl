# ============================================================================
# src/generator/ErrorEstimate/error_estimate_4d.jl
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
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Estimate a ``4``-dimensional tensor-product truncation-error *model* for a composite Newton-Cotes rule
on the hypercube ``[a,b]^4`` using the exact midpoint residual expansion.

# Function description
This routine extends the ``1``-dimensional midpoint-residual model to four dimensions by applying the
``1``-dimensional midpoint error operator along each axis and integrating over the remaining three axes
with tensor-product quadrature.

Let ``\\displaystyle{h = \\frac{b-a}{N}}`` and ``\\displaystyle{\\bar{x} = \\bar{y} = \\bar{z} = \\bar{t} = \\frac{a+b}{2}}``.
From the exact rational composite weights (via midpoint residual moments), we obtain a sequence
of nonzero residual orders ``k`` with exact coefficients
``\\displaystyle{\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}}``. This routine collects the first `nerr_terms`
nonzero residual orders ``k_1, k_2, \\ldots`` (up to `kmax`) and returns the summed separable model:
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i}\\, h^{k_i+1}\\, \\left(
I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} + I_t^{(k_i)}
\\right),
```
where the axis-wise cross integrals are
```math
I_x^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dy\\,dz\\,dt\\;
\\frac{\\partial^k f}{\\partial x^k}(\\bar{x}, y, z, t) \\,,
```
```math
I_y^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dz\\,dt\\,dx\\;
\\frac{\\partial^k f}{\\partial y^k}(x, \\bar{y}, z, t) \\,,
```
```math
I_z^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dt\\,dx\\,dy\\;
\\frac{\\partial^k f}{\\partial z^k}(x, y, \\bar{z}, t) \\,,
```
```math
I_t^{(k)} = \\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b}\\!\\int\\limits_{a}^{b} dx\\,dy\\,dz\\;
\\frac{\\partial^k f}{\\partial t^k}(x, y, z, \\bar{t}) \\,.
```
Each cross-axis integral is evaluated numerically using the same ``1``-dimensional composite nodes/weights
along the remaining axes.

# Arguments

* `f`:
  Scalar callable integrand ``f(x,y,z,t)`` (function, closure, or callable struct).
* `a`, `b`:
  Scalar bounds defining the hypercube ``[a,b]^4``.
* `N`:
  Number of subintervals per axis. Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton–Cotes rule symbol (must be `:ns_pK` style).
* `boundary`:
  Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Keyword arguments

* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model (`1` = LO only, `2` = LO+NLO, ...).
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `Float64`:
  The summed axis-separable truncation-error model value.

# Errors

* Propagates errors from:

  * composite weight assembly / midpoint residual extraction,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
  insufficient nonzero residual terms exist up to `kmax`.

# Notes

* This model sums only *axis-separable* contributions (``x``-only, ``y``-only, ``z``-only, ``t``-only operators).
* Mixed derivative terms (cross terms such as ``\\partial_x^r\\partial_y^s f``) are higher order
  and intentionally omitted.
* Coefficients are derived in exact rational arithmetic and converted to `Float64` only at the final stage.
"""
function error_estimate_4d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol,
    boundary::Symbol;
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

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    # collect LO / LO+NLO / ...
    ks, coeffsR = if nerr_terms == 1
        k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=min(kmax, 64))
        k == 0 && return 0.0
        ([k], Quadrature.RBig[coeffR])
    else
        _leading_midpoint_residual_terms(rule, boundary, N; nterms=nerr_terms, kmax=kmax)
    end

    err = 0.0

    @inbounds for it in eachindex(ks)
        kk = ks[it]
        kk == 0 && continue
        coeff = Float64(coeffsR[it])

        # X-axis
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
                        side=:mid, axis=:x, stage=:midpoint
                    )
                end
            end
        end

        # Y-axis
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
                        side=:mid, axis=:y, stage=:midpoint
                    )
                end
            end
        end

        # Z-axis
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
                        side=:mid, axis=:z, stage=:midpoint
                    )
                end
            end
        end

        # T-axis
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
                        side=:mid, axis=:t, stage=:midpoint
                    )
                end
            end
        end

        err += coeff * h^(kk+1) * (I1 + I2 + I3 + I4)
    end

    return err
end