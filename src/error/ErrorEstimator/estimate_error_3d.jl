# ============================================================================
# src/error/ErrorEstimator/estimate_error_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 3D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_3d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 3D integration error *scale* over the cube domain `[a,b]^3`
using a derivative-based tensor-product heuristic, with optional boundary-
difference handling for selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for stabilizing
convergence fits / extrapolation, not a rigorous truncation bound.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies
a boundary-difference leading-term model **axis by axis**.

For each axis, take a boundary difference of the corresponding axis-wise
derivative and integrate over the remaining coordinates:

- X-axis:
  `I_x = ∬ [∂_x^{dord} f(xL, y, z) - ∂_x^{dord} f(xR, y, z)] dy dz`
- Y-axis:
  `I_y = ∬ [∂_y^{dord} f(x, yL, z) - ∂_y^{dord} f(x, yR, z)] dx dz`
- Z-axis:
  `I_z = ∬ [∂_z^{dord} f(x, y, zL) - ∂_z^{dord} f(x, y, zR)] dx dy`

The final model is

`E ≈ K * h^p * (I_x + I_y + I_z)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `y`, `z`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This boundary-difference structure is designed to reflect the leading behavior
of certain endpoint-free chained formulas and improve χ² stability.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine follows the legacy midpoint tensor heuristic:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate three axis-wise contributions by integrating the `m`-th derivative
  along one axis while fixing the remaining coordinates at quadrature nodes:
  - `I_x = ∬ ∂_x^m f(x̄, y, z) dy dz`
  - `I_y = ∬ ∂_y^m f(x, ȳ, z) dx dz`
  - `I_z = ∬ ∂_z^m f(x, y, z̄) dx dy`
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y + I_z)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 3D integrand callable `f(x, y, z)` (function, closure, or callable struct).
- `a`, `b`: Cube domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Integration rule symbol.

# Returns
- `Float64`: heuristic signed error estimate (scale model). If `rule` is not
  recognized (and no boundary model is defined), returns `0.0`.

# Notes
- This estimator intentionally matches the loop structure and accumulation style
  of the original implementation for reproducibility.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path, and
  - enforced explicitly in the boundary-model branch for supported open rules.
"""
function estimate_error_3d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx

    # ---- special boundary-difference models ----
    if _has_boundary_error_model(rule)
        if rule == :simpson13_open
            (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
            (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")
        elseif rule == :bode_open
            (N % 4 == 0) || error("Open composite Boole requires N divisible by 4, got N = $N")
            (N >= 16)    || error("Open composite Boole requires N ≥ 16 (non-overlapping end stencils), got N = $N")
        end

        p, K, dord, off = _boundary_error_params(rule)
        xL = aa + off*h
        xR = aa + (N-off)*h
        yL = xL; yR = xR
        zL = xL; zR = xR

        # X-axis: ∬ [∂_x^(dord) f(xL,y,z) - ∂_x^(dord) f(xR,y,z)] dy dz
        I1 = 0.0
        @inbounds for j in eachindex(ys)
            y = ys[j]
            wyj = wy[j]
            for k in eachindex(zs)
                z = zs[k]
                gx(x) = f(x, y, z)
                I1 += wyj * wz[k] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
            end
        end

        # Y-axis
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k in eachindex(zs)
                z = zs[k]
                gy(y) = f(x, y, z)
                I2 += wxi * wz[k] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
            end
        end

        # Z-axis
        I3 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                gz(z) = f(x, y, z)
                I3 += wxi * wy[j] * (nth_derivative(gz, zL, dord) - nth_derivative(gz, zR, dord))
            end
        end

        return K * h^p * (I1 + I2 + I3)
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    I1 = 0.0
    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k in eachindex(zs)
            z = zs[k]
            gx(x) = f(x, y, z)
            I1 += wyj * wz[k] * nth_derivative(gx, x̄, m)
        end
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k in eachindex(zs)
            z = zs[k]
            gy(y) = f(x, y, z)
            I2 += wxi * wz[k] * nth_derivative(gy, ȳ, m)
        end
    end

    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            gz(z) = f(x, y, z)
            I3 += wxi * wy[j] * nth_derivative(gz, z̄, m)
        end
    end

    return C*(bb-aa)*h^m*(I1 + I2 + I3)
end