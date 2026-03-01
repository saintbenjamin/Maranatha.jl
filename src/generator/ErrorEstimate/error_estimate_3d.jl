# ============================================================================
# src/generator/ErrorEstimate/error_estimate_3d.jl
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
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

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

# Arguments

* `f`:
  Scalar callable integrand ``f(x,y,z)`` (function, closure, or callable struct).
* `a`, `b`:
  Scalar bounds defining the cube domain ``[a,b]^3``.
* `N`:
  Number of subintervals per axis. Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
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

  * midpoint residual extraction / composite weight generation,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
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

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

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

        # X-axis contribution: d^kk/dx^kk then integrate over y,z
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
                    side=:mid, axis=:x, stage=:midpoint
                )
            end
        end

        # Y-axis contribution: d^kk/dy^kk then integrate over x,z
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
                    side=:mid, axis=:y, stage=:midpoint
                )
            end
        end

        # Z-axis contribution: d^kk/dz^kk then integrate over x,y
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
                    side=:mid, axis=:z, stage=:midpoint
                )
            end
        end

        err += coeff * h^(kk+1) * (I1 + I2 + I3)
    end

    return err
end