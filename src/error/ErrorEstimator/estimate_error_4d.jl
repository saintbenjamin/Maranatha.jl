# ============================================================================
# src/error/ErrorEstimator/estimate_error_4d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_4d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    ) -> Float64

Estimate the leading tensor-product truncation error for a ``4``-dimensional composite Newton-Cotes rule
on the hypercube ``[a,b]^4`` using the exact midpoint residual expansion.

# Function description
This routine extends the ``1``-dimensional midpoint residual model to four dimensions by applying
the ``1``-dimensional midpoint error operator along each axis and integrating over the remaining
three axes using tensor-product quadrature.

Let ``\\displaystyle{h = \\frac{b-a}{N}}``.
The leading midpoint residual order ``k`` and its coefficient
`coeffR` are extracted from the exact rational composite weights for the ``1``-dimensional rule
associated with `(rule, boundary, N)`.

The returned model is:
```math
E = \\texttt{coeff} \\, h^{k+1} \\, \\left( I_x + I_y + I_z + I_t \\right)
```
with `coeff = Float64(coeffR)` and midpoint coordinates:
```math
\\bar{x} = \\bar{y} = \\bar{z} = \\bar{t} = \\frac{a+b}{2} \\,.
```
The axis contributions are:
- ``\\displaystyle{I_x = \\int\\int\\int dy dz dt \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y , z , t \\right)}``
- ``\\displaystyle{I_y = \\int\\int\\int dz dt dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} , z , t \\right)}``
- ``\\displaystyle{I_z = \\int\\int\\int dt dx dy \\; \\frac{\\partial^k f}{\\partial z^k} \\left( x , y , \\bar{z} , t \\right)}``
- ``\\displaystyle{I_t = \\int\\int\\int dx dy dz \\; \\frac{\\partial^k f}{\\partial t^k} \\left( x , y , z , \\bar{t} \\right)}``
Each integral is computed numerically using the same ``1``-dimensional quadrature nodes/weights.

# Mathematical structure
For tensor-product quadrature on ``[a,b]^4`` the leading separable error model is:
```math
E = C_k \\, h^{k+1} \\left[ 
\\int\\int\\int dy dz dt \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y , z , t \\right) 
+  
\\int\\int\\int dz dt dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} , z , t \\right) 
+  
\\int\\int\\int dt dx dy \\; \\frac{\\partial^k f}{\\partial z^k} \\left( x , y , \\bar{z} , t \\right) 
+  
\\int\\int\\int dx dy dz \\; \\frac{\\partial^k f}{\\partial t^k} \\left( x , y , z , \\bar{t} \\right) 
\\right] 
+ \\left( \\text{higher-order terms} \\right) \\,.
```
Higher-order residual terms and mixed-derivative corrections are omitted.

# Arguments
- `f`:
    Scalar callable integrand ``f(x,y,z,t)`` (function, closure, or callable struct).
- `a`, `b`:
    Scalar bounds defining the hypercube ``[a,b]^4``.
- `N`:
    Number of subintervals per axis (must satisfy composite constraints for `(rule, boundary)`).
- `rule`:
    Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
- `boundary`:
    Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Returns
- `Float64`:
    Leading tensor-product truncation error estimate.

# Errors
- Propagates any errors from:
  - composite weight assembly,
  - midpoint residual extraction,
  - derivative evaluation ([`nth_derivative`](@ref)).

# Notes
- This routine intentionally returns a *leading-term error scale*.
- Coefficients come from exact rational arithmetic and are converted to `Float64`
  only at the final stage.
- Returns `0.0` if the residual scan reports `k == 0` (degenerate/unexpected case).
"""
function estimate_error_4d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol,
    boundary::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2
    t̄ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    # ------------------------------------------------------------
    # Default tensor-style midpoint model (auto from β residual)
    # ------------------------------------------------------------

    k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=64)
    k == 0 && return 0.0
    coeff = Float64(coeffR)

    # X-axis
    I1 = 0.0
    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k2 in eachindex(zs)
            z = zs[k2]
            wyj_wzk = wyj * wz[k2]
            for l in eachindex(ts)
                t = ts[l]
                gx(x) = f(x, y, z, t)
                I1 += wyj_wzk * wt[l] * nth_derivative(
                    gx, x̄, k;
                    h=h, rule=rule, N=N, dim=4,
                    side=:mid, axis=:x, stage=:midpoint
                )
            end
        end
    end

    # Y-axis
    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k2 in eachindex(zs)
            z = zs[k2]
            wxi_wzk = wxi * wz[k2]
            for l in eachindex(ts)
                t = ts[l]
                gy(y) = f(x, y, z, t)
                I2 += wxi_wzk * wt[l] * nth_derivative(
                    gy, ȳ, k;
                    h=h, rule=rule, N=N, dim=4,
                    side=:mid, axis=:y, stage=:midpoint
                )
            end
        end
    end

    # Z-axis
    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for l in eachindex(ts)
                t = ts[l]
                gz(z) = f(x, y, z, t)
                I3 += wxi_wyj * wt[l] * nth_derivative(
                    gz, z̄, k;
                    h=h, rule=rule, N=N, dim=4,
                    side=:mid, axis=:z, stage=:midpoint
                )
            end
        end
    end

    # T-axis
    I4 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for k2 in eachindex(zs)
                z = zs[k2]
                gt(t) = f(x, y, z, t)
                I4 += wxi_wyj * wz[k2] * nth_derivative(
                    gt, t̄, k;
                    h=h, rule=rule, N=N, dim=4,
                    side=:mid, axis=:t, stage=:midpoint
                )
            end
        end
    end

    return coeff * h^(k+1) * (I1 + I2 + I3 + I4)
end
