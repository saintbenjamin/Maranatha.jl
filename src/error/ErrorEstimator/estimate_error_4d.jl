# ============================================================================
# src/error/ErrorEstimator/estimate_error_4d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 4D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_4d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 4D integration error *scale* over the hypercube domain `[a,b]^4`
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
derivative and integrate over the remaining three coordinates:

- X-axis:
  `I_x = ∭ [∂_x^{dord} f(xL, y, z, t) - ∂_x^{dord} f(xR, y, z, t)] dy dz dt`
- Y-axis:
  `I_y = ∭ [∂_y^{dord} f(x, yL, z, t) - ∂_y^{dord} f(x, yR, z, t)] dx dz dt`
- Z-axis:
  `I_z = ∭ [∂_z^{dord} f(x, y, zL, t) - ∂_z^{dord} f(x, y, zR, t)] dx dy dt`
- T-axis:
  `I_t = ∭ [∂_t^{dord} f(x, y, z, tL) - ∂_t^{dord} f(x, y, z, tR)] dx dy dz`

The final model is

`E ≈ K * h^p * (I_x + I_y + I_z + I_t)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `y`, `z`, `t`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This boundary-difference structure is designed to reflect the leading behavior
of certain endpoint-free chained formulas and improve χ² stability.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine follows the legacy midpoint tensor heuristic:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate four axis-wise contributions by integrating the `m`-th derivative
  along one axis while fixing the remaining coordinates at quadrature nodes:
  - `I_x = ∭ ∂_x^m f(x̄, y, z, t) dy dz dt`
  - `I_y = ∭ ∂_y^m f(x, ȳ, z, t) dx dz dt`
  - `I_z = ∭ ∂_z^m f(x, y, z̄, t) dx dy dt`
  - `I_t = ∭ ∂_t^m f(x, y, z, t̄) dx dy dz`
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y + I_z + I_t)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 4D integrand callable `f(x, y, z, t)` (function, closure, or callable struct).
- `a`, `b`: Hypercube domain bounds.
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
function estimate_error_4d(
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
    t̄ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

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
        tL = xL; tR = xR

        # X-axis
        I1 = 0.0
        @inbounds for j in eachindex(ys)
            y = ys[j]
            wyj = wy[j]
            for k in eachindex(zs)
                z = zs[k]
                wyj_wzk = wyj * wz[k]
                for l in eachindex(ts)
                    t = ts[l]
                    gx(x) = f(x, y, z, t)
                    I1 += wyj_wzk * wt[l] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
                end
            end
        end

        # Y-axis
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k in eachindex(zs)
                z = zs[k]
                wxi_wzk = wxi * wz[k]
                for l in eachindex(ts)
                    t = ts[l]
                    gy(y) = f(x, y, z, t)
                    I2 += wxi_wzk * wt[l] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
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
                    I3 += wxi_wyj * wt[l] * (nth_derivative(gz, zL, dord) - nth_derivative(gz, zR, dord))
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
                for k in eachindex(zs)
                    z = zs[k]
                    gt(t) = f(x, y, z, t)
                    I4 += wxi_wyj * wz[k] * (nth_derivative(gt, tL, dord) - nth_derivative(gt, tR, dord))
                end
            end
        end

        return K * h^p * (I1 + I2 + I3 + I4)
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
            wyj_wzk = wyj * wz[k]
            for l in eachindex(ts)
                t = ts[l]
                gx(x) = f(x, y, z, t)
                I1 += wyj_wzk * wt[l] * nth_derivative(gx, x̄, m)
            end
        end
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k in eachindex(zs)
            z = zs[k]
            wxi_wzk = wxi * wz[k]
            for l in eachindex(ts)
                t = ts[l]
                gy(y) = f(x, y, z, t)
                I2 += wxi_wzk * wt[l] * nth_derivative(gy, ȳ, m)
            end
        end
    end

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
                I3 += wxi_wyj * wt[l] * nth_derivative(gz, z̄, m)
            end
        end
    end

    I4 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for k in eachindex(zs)
                z = zs[k]
                gt(t) = f(x, y, z, t)
                I4 += wxi_wyj * wz[k] * nth_derivative(gt, t̄, m)
            end
        end
    end

    return C*(bb-aa)*h^m*(I1 + I2 + I3 + I4)
end
