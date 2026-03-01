# ============================================================================
# src/error/ErrorEstimator/estimate_error_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_3d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    ) -> Float64

Estimate the leading tensor-product truncation error for a ``3``-dimensional composite Newton-Cotes rule
on the cube ``[a,b]^3`` using the exact midpoint residual expansion.

# Function description
This routine generalizes the ``1``-dimensional midpoint residual model to a ``3``-dimensional tensor-product setting
by applying the ``1``*-dimensional midpoint error operator* along each axis and numerically integrating
the resulting derivative across the remaining axes.

Let ``\\displaystyle{h = \\frac{b-a}{N}}``. 
From the exact rational composite weights associated with
`(rule, boundary, N)`, the code extracts:

- the leading nonzero residual order ``k``, and
- the corresponding exact rational coefficient `coeffR`.

The modeled leading truncation error is:
```math
E = \\texttt{coeff} \\, h^{k+1} \\, \\left( I_x + I_y + I_z \\right)
```
where:
- `coeff = Float64(coeffR)`,
- ``\\displaystyle{I_x = \\int\\int dy dz \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y , z \\right)}``
- ``\\displaystyle{I_y = \\int\\int dz dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} , z \\right)}``
- ``\\displaystyle{I_z = \\int\\int dx dy \\; \\frac{\\partial^k f}{\\partial z^k} \\left( x , y , \\bar{z} \\right)}``
and ``\\displaystyle{\\bar{x} = \\bar{y} = \\bar{z} = \\frac{a+b}{2}}``.
Each cross-axis integral is computed using the same ``1``-dimensional quadrature nodes/weights
along the other axes.

# Mathematical structure
For a tensor-product quadrature on ``[a,b]^3`` the leading error contribution
separates into axis-wise terms:
```math
E = C_k \\, h^{k+1} \\left[ 
\\int\\int dy dz \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y , z \\right) 
+  
\\int\\int dz dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} , z \\right) 
+  
\\int\\int dx dy \\; \\frac{\\partial^k f}{\\partial z^k} \\left( x , y , \\bar{z} \\right) 
\\right] 
+ \\left( \\text{higher-order terms} \\right) \\,.
```
Mixed-derivative contributions and higher residual orders are not included in this
leading model.

# Arguments
- `f`:
    Scalar callable integrand ``f(x,y,z)`` (function, closure, or callable struct).
- `a`, `b`:
    Scalar bounds defining the cube domain ``[a,b]^3``.
- `N`:
    Number of subintervals per axis.
    Must satisfy the composite tiling constraint for `(rule, boundary)`.
- `rule`:
    Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
- `boundary`:
    Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Returns
- `Float64`:
    Leading tensor-product truncation error estimate.

# Errors
- Propagates any errors from:
  - midpoint residual extraction,
  - composite weight generation,
  - derivative evaluation ([`nth_derivative`](@ref)).

# Notes
- This is a *leading-term asymptotic model*, not a rigorous bound.
- Coefficients come from exact rational arithmetic and are converted to `Float64`
  only at the final stage.
- Returns `0.0` if the residual scan reports `k == 0` (degenerate/unexpected case).
"""
function estimate_error_3d(
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

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

    # ------------------------------------------------------------
    # Default tensor-style midpoint model (auto from β residual)
    # ------------------------------------------------------------

    k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=64)
    k == 0 && return 0.0
    coeff = Float64(coeffR)

    # X-axis contribution
    I1 = 0.0
    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k2 in eachindex(zs)
            z = zs[k2]
            gx(x) = f(x, y, z)
            I1 += wyj * wz[k2] * nth_derivative(
                gx, x̄, k;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:x, stage=:midpoint
            )
        end
    end

    # Y-axis contribution
    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k2 in eachindex(zs)
            z = zs[k2]
            gy(y) = f(x, y, z)
            I2 += wxi * wz[k2] * nth_derivative(
                gy, ȳ, k;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:y, stage=:midpoint
            )
        end
    end

    # Z-axis contribution
    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            gz(z) = f(x, y, z)
            I3 += wxi * wy[j] * nth_derivative(
                gz, z̄, k;
                h=h, rule=rule, N=N, dim=3,
                side=:mid, axis=:z, stage=:midpoint
            )
        end
    end

    return coeff * h^(k+1) * (I1 + I2 + I3)
end