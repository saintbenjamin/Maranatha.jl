# ============================================================================
# src/error/ErrorEstimator/estimate_error_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 2D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_2d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 2D integration error *scale* over the square domain `[a,b] × [a,b]`
using a derivative-based tensor-product heuristic, with optional boundary-
difference handling for selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for stabilizing
convergence fits / extrapolation, not a rigorous truncation bound.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies
a boundary-difference leading-term model **axis by axis**:

- Along the `x`-axis, integrate over `y`:
  `I_x = ∫ [∂_x^{dord} f(xL, y) - ∂_x^{dord} f(xR, y)] dy`
- Along the `y`-axis, integrate over `x`:
  `I_y = ∫ [∂_y^{dord} f(x, yL) - ∂_y^{dord} f(x, yR)] dx`

The final model is

`E ≈ K * h^p * (I_x + I_y)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `yL, yR`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This reflects the boundary-dominant leading behavior of certain endpoint-free
chained rules, and is used to improve stability in χ²-based convergence fits.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine uses the same tensor-product “single-sample midpoint”
structure as the legacy estimator:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate two axis-wise contributions:
  1) `I_x = ∫ ∂_x^m f(x̄, y) dy` by sampling `∂_x^m` at `x̄` and integrating over `y`.
  2) `I_y = ∫ ∂_y^m f(x, ȳ) dx` by sampling `∂_y^m` at `ȳ` and integrating over `x`.
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 2D integrand callable `f(x, y)` (function, closure, or callable struct).
- `a`, `b`: Square domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Integration rule symbol (same family as in `estimate_error_1d`).

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
function estimate_error_2d(
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

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)

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
        yL = xL
        yR = xR

        # X-axis boundary difference, integrated over y
        I1 = 0.0
        @inbounds for j in eachindex(xs)
            y = xs[j]
            gx(x) = f(x, y)
            I1 += wx[j] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
        end

        # Y-axis boundary difference, integrated over x
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            gy(y) = f(x, y)
            I2 += wx[i] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
        end

        return K * h^p * (I1 + I2)
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    I1 = 0.0
    @inbounds for j in eachindex(xs)
        y = xs[j]
        gx(x) = f(x,y)
        I1 += wx[j]*nth_derivative(gx, x̄, m)
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        gy(y) = f(x,y)
        I2 += wx[i]*nth_derivative(gy, ȳ, m)
    end

    return C*(bb-aa)*h^m*(I1 + I2)
end
