# ============================================================================
# src/error/ErrorEstimator/estimate_error_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_2d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    ) -> Float64

Estimate the leading tensor-product truncation error for a ``2``-dimensional composite
Newton-Cotes rule on the square domain ``[a,b]^2``.

# Function description
This routine extends the ``1``-dimensional midpoint residual model axis-by-axis
to tensor-product quadrature.

Let ``\\displaystyle{h = \\frac{b-a}{N}}``.
From the exact ``1``-dimensional composite rule, determine the leading midpoint residual order ``k``
and its exact coefficient `coeff`.

In two dimensions, the tensor-product truncation error decomposes
into axis-wise contributions:
```math
E = \\texttt{coeff} \\, h^{k+1} \\, \\left( I_x + I_y \\right)
```
where:

- ``\\displaystyle{I_x = \\int dy \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y \\right)}``
- ``\\displaystyle{I_y = \\int dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} \\right)}``
with midpoint coordinates:
```math
\\bar{x} = \\bar{y} = \\frac{a+b}{2}
```
The cross-axis integrals are evaluated numerically using the same
composite quadrature weights.

# Mathematical structure
For tensor-product rules, the leading truncation error separates as:
```math
E = C_k \\, h^{k+1} \\left[ 
\\int dy \\; \\frac{\\partial^k f}{\\partial x^k} \\left( \\bar{x} , y \\right) 
+ 
\\int dx \\; \\frac{\\partial^k f}{\\partial y^k} \\left( x , \\bar{y} \\right) 
\\right] 
+ \\left( \\text{higher-order terms} \\right) \\,.
```
Mixed derivative contributions appear at higher asymptotic order
and are not included in this leading model.

# Arguments
- `f`:
    Scalar callable integrand ``f(x,y)`` (function, closure, or callable struct).
- `a`, `b`:
    Scalar bounds defining the square domain ``[a,b]^2``.
- `N`:
    Number of subintervals per axis.
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
- This routine models only the leading separable contribution.
- Mixed derivative terms of the form ``\\displaystyle{\\frac{\\partial^r \\partial^s f}{\\partial x^r \\partial y^s}}`` appear at higher orders and are intentionally omitted.
- Coefficients originate from exact rational composite weights.
"""
function estimate_error_2d(
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

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=64)
    k == 0 && return 0.0
    coeff = Float64(coeffR)

    # X-axis contribution: apply 1D error operator in x, integrate over y
    I1 = 0.0
    @inbounds for j in eachindex(xs)
        y = xs[j]
        gx(x) = f(x, y)
        I1 += wx[j] * nth_derivative(
            gx, x̄, k;
            h=h, rule=rule, N=N, dim=2,
            side=:mid, axis=:x, stage=:midpoint
        )
    end

    # Y-axis contribution: apply 1D error operator in y, integrate over x
    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        gy(y) = f(x, y)
        I2 += wx[i] * nth_derivative(
            gy, ȳ, k;
            h=h, rule=rule, N=N, dim=2,
            side=:mid, axis=:y, stage=:midpoint
        )
    end

    # Each axis contributes coeff*h^(k+1) times the cross-axis integral
    return coeff * h^(k+1) * (I1 + I2)
end