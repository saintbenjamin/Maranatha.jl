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

Return a fast 3D integration error *scale* proxy for the cube integral
`∫_a^b ∫_a^b ∫_a^b f(x,y,z) dx dy dz` over `[a,b]^3`, using a lightweight
derivative-based tensor-product heuristic.

For selected endpoint-free (“open-chain”) rules, this routine switches to an
axis-wise boundary-difference proxy that is often more stable in χ²-based
convergence fits. All derivatives are attempted with ForwardDiff first, with an
automatic Taylor fallback when the ForwardDiff result is non-finite (`Inf`/`NaN`).

# Function description
This routine provides a **cheap, consistent error scale model** intended to:
- supply per-point weights `σ(h)` for χ²-based convergence fits (`fit_convergence`), and
- match the same `h`-scaling convention used by the 1D/2D/4D estimators in this package.

It is **not** a rigorous truncation bound and does not attempt to reproduce the
full composite-rule error expansion.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies an
axis-wise boundary-difference proxy. For each axis, it takes a boundary difference
of the corresponding axis-wise derivative and integrates over the remaining coordinates:

- X-axis:
  `I_x ≈ ∬ [∂_x^{dord} f(xL, y, z) - ∂_x^{dord} f(xR, y, z)] dy dz`
- Y-axis:
  `I_y ≈ ∬ [∂_y^{dord} f(x, yL, z) - ∂_y^{dord} f(x, yR, z)] dx dz`
- Z-axis:
  `I_z ≈ ∬ [∂_z^{dord} f(x, y, zL) - ∂_z^{dord} f(x, y, zR)] dx dy`

and returns

`E ≈ K * h^p * (I_x + I_y + I_z)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `y`, `z`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This branch is designed for endpoint-free chained formulas whose leading error
behavior can be boundary-dominant, and empirically improves χ² stability for
those rules.

### Derivative evaluation and Taylor fallback
All derivatives in this routine are evaluated via the internal helper `_nth_deriv_safe`:
1) compute using `nth_derivative` (ForwardDiff-based),
2) if non-finite, emit a `warn_benji` and retry with `nth_derivative_taylor` (TaylorSeries-based),
3) throw an error only if the Taylor fallback is also non-finite.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, the estimator follows the same “single-sample midpoint derivative” pattern
used throughout the tensor-product estimators:

1) Obtain `(m, C)` via `_rule_params_for_tensor_error(rule)`.
2) Build 1D nodes/weights `(xs, wx)` via `quadrature_1d_nodes_weights(a, b, N, rule)`.
3) Approximate three axis-wise contributions by integrating the `m`-th derivative
   along one axis while holding the remaining coordinates on quadrature nodes:
   - `I_x = ∬ ∂_x^m f(x̄, y, z) dy dz`
   - `I_y = ∬ ∂_y^m f(x, ȳ, z) dx dz`
   - `I_z = ∬ ∂_z^m f(x, y, z̄) dx dy`
4) Return

`E ≈ C * (b-a) * h^m * (I_x + I_y + I_z)`.

# Arguments
- `f`: 3D integrand callable `f(x, y, z)` (function, closure, or callable struct).
- `a`, `b`: Cube domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Quadrature rule symbol.

# Returns
- `Float64`: A heuristic (signed) error scale proxy. If `m == 0` for the selected
  `rule`, returns `0.0`.

# Notes
- Some open-chain rules may have **negative quadrature weights**. This estimator
  intentionally preserves the rule-defined weights rather than enforcing normalization.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced explicitly in the boundary-model branch for supported open rules, and
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path.
- The Taylor fallback requires the integrand to accept generic number types
  (e.g. `Taylor1`). If the integrand dispatch is restricted to `Real` only, the fallback
  may raise a `MethodError`.

# Errors
- Throws an error if `(N, rule)` violates rule constraints.
- Throws an error if both ForwardDiff and Taylor derivatives are non-finite in the
  selected estimator branch.
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

    @inline function _nth_deriv_safe(g, x, n; side::Symbol=:mid, axis::Symbol=:x)
        d = nth_derivative(g, x, n)
        if !isfinite(d)
            JobLoggerTools.warn_benji(
                "Non-finite derivative (ForwardDiff); trying Taylor fallback " *
                "h=$h x=$x n=$n rule=$rule N=$N side=$side axis=$axis"
            )
            d = nth_derivative_taylor(g, x, n)
            if !isfinite(d)
                JobLoggerTools.error_benji(
                    "Non-finite in 3D error estimator even after Taylor fallback: " *
                    "h=$h x=$x deriv=$d n=$n rule=$rule N=$N side=$side axis=$axis"
                )
            end
        end
        return d
    end

    # ---- special boundary-difference models ----
    if _has_boundary_error_model(rule)
        if rule == :simpson13_open
            (N % 2 == 0) || JobLoggerTools.error_benji("open composite Simpson 1/3 rule requires N even, got N = $N")
            (N >= 8)     || JobLoggerTools.error_benji("open composite Simpson 1/3 rule requires N ≥ 8, got N = $N")
        elseif rule == :bode_open
            (N % 4 == 0) || JobLoggerTools.error_benji("Open composite Boole's rule requires N divisible by 4, got N = $N")
            (N >= 16)    || JobLoggerTools.error_benji("Open composite Boole's rule requires N ≥ 16 (non-overlapping end stencils), got N = $N")
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
                # I1 += wyj * wz[k] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
                I1 += wyj * wz[k] * (
                    _nth_deriv_safe(gx, xL, dord; side=:L, axis=:x) -
                    _nth_deriv_safe(gx, xR, dord; side=:R, axis=:x)
                )
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
                # I2 += wxi * wz[k] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
                I2 += wxi * wz[k] * (
                    _nth_deriv_safe(gy, yL, dord; side=:L, axis=:y) -
                    _nth_deriv_safe(gy, yR, dord; side=:R, axis=:y)
                )
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
                # I3 += wxi * wy[j] * (nth_derivative(gz, zL, dord) - nth_derivative(gz, zR, dord))
                I3 += wxi * wy[j] * (
                    _nth_deriv_safe(gz, zL, dord; side=:L, axis=:z) -
                    _nth_deriv_safe(gz, zR, dord; side=:R, axis=:z)
                )
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
            # I1 += wyj * wz[k] * nth_derivative(gx, x̄, m)
            I1 += wyj * wz[k] * _nth_deriv_safe(gx, x̄, m; side=:mid, axis=:x)
        end
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k in eachindex(zs)
            z = zs[k]
            gy(y) = f(x, y, z)
            # I2 += wxi * wz[k] * nth_derivative(gy, ȳ, m)
            I2 += wxi * wz[k] * _nth_deriv_safe(gy, ȳ, m; side=:mid, axis=:y)
        end
    end

    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            gz(z) = f(x, y, z)
            # I3 += wxi * wy[j] * nth_derivative(gz, z̄, m)
            I3 += wxi * wy[j] * _nth_deriv_safe(gz, z̄, m; side=:mid, axis=:z)
        end
    end

    return C*(bb-aa)*h^m*(I1 + I2 + I3)
end