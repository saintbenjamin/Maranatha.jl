# ============================================================================
# src/error/ErrorEstimator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorEstimator

using ForwardDiff
using LinearAlgebra
using ..Integrate
using ..BodeRule_MinOpen_MaxOpen

export estimate_error

# ============================================================
# Internal helpers (must preserve numerical behavior)
# ============================================================

"""
    nth_derivative(f, x::Real, n::Int)

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using repeated `ForwardDiff.derivative`.

# Function description
This routine is intentionally written to accept any **callable** object `f`,
not only subtypes of `Function`. This includes:
- ordinary functions,
- anonymous closures,
- callable structs (functors) such as preset integrands.

This design is required for compatibility with the integrand registry and
preset-style callable wrappers while preserving ForwardDiff-based behavior.

# Arguments
- `f`: Scalar-to-scalar callable (e.g., `f(x)::Number`).
- `x::Real`: Point at which the derivative is evaluated.
- `n::Int`: Derivative order (nonnegative integer).

# Returns
- The `n`-th derivative value `f^(n)(x)`.

# Notes
- This implementation constructs a nested closure chain of length `n` and then
  evaluates it at `x`. This intentionally matches the original behavior.
- Type restriction `f::Function` is intentionally avoided because callable
  structs are not subtypes of `Function`, but must be supported.
"""
function nth_derivative(f, x::Real, n::Int)
    g = f
    for _ in 1:n
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
    end
    return g(x)
end

"""
    _rule_params_for_tensor_error(rule::Symbol)

Map `rule` to the derivative order `m` and coefficient `C` used by the
tensor-product derivative-based error heuristics in 2D/3D/4D estimators.

# Arguments
- `rule`: Integration rule symbol.

# Returns
- `(m, C)` where:
  - `m::Int` is the derivative order used in the estimator,
  - `C` is the rule-dependent coefficient (kept as the same literal type
    as the original implementation).

If `rule` is not supported, returns `(0, 0.0)`.
"""
function _rule_params_for_tensor_error(rule::Symbol)
    # IMPORTANT: keep the exact literals/types consistent with the original code.
    if rule == :simpson13_close
        return (4, -1/180)
    elseif rule == :simpson38_close
        return (4, -1/80)
    elseif rule == :bode_close
        return (6, -2/945)
    elseif rule == :simpson13_open
        return (3, -3/8)
    elseif rule == :simpson38_open
        return (4, 14/45)
    elseif rule == :bode_open
        return (6, 1.0)
    else
        return (0, 0.0)
    end
end

# ============================================================
# 1D error estimator
# ============================================================

"""
    estimate_error_1d(f, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate the integration error for a 1D integral of `f(x)` over `[a, b]`
using Newton–Cotes quadrature rules and automatic differentiation.

# Function description
This function provides rule-specific, derivative-based error models.
Some rules use a single midpoint derivative (heuristic leading term),
while others use endpoint differences or panel-wise derivative expansions,
matching the original implementation.

# Arguments
- `f`: Integrand callable `f(x)` (function, closure, or callable struct).
- `a`, `b`: Integration limits (scalars).
- `N`: Number of subintervals (must satisfy rule-specific divisibility and minimum constraints).
- `rule`: Integration rule symbol:
  - `:simpson13_close` → closed composite Simpson 1/3 (midpoint 4th-derivative leading-term heuristic)
  - `:simpson38_close` → closed composite Simpson 3/8 (midpoint 4th-derivative leading-term heuristic)
  - `:bode_close`      → closed composite Bode (midpoint 6th-derivative leading-term heuristic)
  - `:simpson13_open`  → open-chain Simpson 1/3 (endpoint 3rd-derivative difference model)
  - `:simpson38_open`  → open chained 3-point Newton–Cotes (panel-wise 4th/6th/8th derivative expansion)
  - `:bode_open`       → open-chain Bode (delegates to `BodeRule_MinOpen_MaxOpen`)

# Returns
- A `Float64` estimate of the total integration error (as returned by the original rule-specific formula).
  If `rule` is not recognized, returns `0.0`.

# Errors
- Throws an error if `N` violates rule-specific constraints.
"""
function estimate_error_1d(f, a::Real, b::Real, N::Int, rule::Symbol)
    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xj(j::Int) = aa + j * h

    if rule == :simpson13_close
        # closed composite Simpson 1/3 (leading term heuristic)
        N % 2 == 0 || error("Simpson 1/3 requires N divisible by 2, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 180.0) * h^4 * d4

    elseif rule == :simpson38_close
        # closed composite Simpson 3/8 (leading term heuristic)
        N % 3 == 0 || error("Simpson 3/8 requires N divisible by 3, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 80.0) * h^4 * d4

    elseif rule == :bode_close
        # closed composite Bode (leading term heuristic)
        N % 4 == 0 || error("Bode's rule requires N divisible by 4, got N = $N")
        x̄ = (aa + bb) / 2
        d6 = nth_derivative(f, x̄, 6)
        return -((2.0 / 945.0) * (bb - aa)) * h^6 * d6

    # ----------------------------
    # Open-chain (your definitions)
    # ----------------------------

    elseif rule == :simpson13_open
        (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
        (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")

        # Leading-term error model consistent with the open-chain expansion:
        #   E ≈ -(3/8) h^4 [ f'''( (x1+x2)/2 ) - f'''( (x_{N-2}+x_{N-1})/2 ) ]
        #
        # Here E is an estimate of (exact - quadrature).

        xL  = aa + 1.5 * h           # (x1 + x2)/2
        xR  = aa + (N - 1.5) * h     # (x_{N-2} + x_{N-1})/2

        d3L = nth_derivative(f, xL, 3)
        d3R = nth_derivative(f, xR, 3)

        return -(3.0 / 8.0) * h^4 * (d3L - d3R)

    elseif rule == :simpson38_open
        # IMPORTANT:
        # This is NOT the classical Simpson 3/8 composite rule.
        # This is the endpoint-free open 3-point Newton–Cotes chained rule (panel width = 4h).
        N % 4 == 0 || error("Open 3-point chained rule requires N divisible by 4, got N = $N")
        N >= 4     || error("Open 3-point chained rule requires N ≥ 4, got N = $N")

        # Panel centers: x_{4k+2}, k = 0..N/4-1
        err = 0.0
        M = N ÷ 4

        for k in 0:(M-1)
            xc = xj(4k + 2)
            err += (14.0 / 45.0)    * h^5 * nth_derivative(f, xc, 4)
            err += (41.0 / 945.0)   * h^7 * nth_derivative(f, xc, 6)
            err += (61.0 / 22680.0) * h^9 * nth_derivative(f, xc, 8)
        end

        return err

    elseif rule == :bode_open
        (N % 4 == 0) || error("Bode open-chain (open composite Boole) requires N divisible by 4, got N = $N")
        (N >= 16)    || error("Bode open-chain (open composite Boole) requires N ≥ 16, got N = $N")

        err = BodeRule_MinOpen_MaxOpen.bode_rule_min_open_max_open_error(f, aa, bb, N; nth_derivative=nth_derivative)

        return err

    else
        return 0.0
    end
end

# ============================================================
# 2D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_2d(f, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate integration error for a 2D integral over a square domain `[a,b] × [a,b]`
using a derivative-based tensor-product heuristic.

# Function description
For supported rules, this estimator:
1) Builds the 1D quadrature nodes/weights for the given `rule`.
2) Approximates the axis-wise contribution by integrating the `m`-th derivative
   along one axis while fixing the other axis at the midpoint, and sums both axes.

This matches the original implementation exactly, including loop ordering and
floating-point accumulation order.

# Arguments
- `f`: 2D integrand callable `f(x, y)` (function, closure, or callable struct).
- `a`, `b`: Square domain bounds.
- `N`: Number of subdivisions per axis.
- `rule`: Integration rule symbol (same set as in `estimate_error_1d`).

# Returns
- A `Float64` error estimate. If `rule` is not recognized, returns `0.0`.
"""
function estimate_error_2d(f, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2

    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)

    # ---- term 1 : integrate ∂^m/∂x^m f(x̄,y) dy
    I1 = 0.0
    for j in eachindex(xs)
        y = xs[j]
        gx(x) = f(x,y)
        I1 += wx[j]*nth_derivative(gx, x̄, m)
    end

    # ---- term 2 : integrate ∂^m/∂y^m f(x,ȳ) dx
    I2 = 0.0
    for i in eachindex(xs)
        x = xs[i]
        gy(y) = f(x,y)
        I2 += wx[i]*nth_derivative(gy, ȳ, m)
    end

    return C*(bb-aa)*h^m*(I1 + I2)
end

# ============================================================
# 3D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_3d(f, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate integration error for a 3D integral over a cube domain `[a,b]^3`
using a derivative-based tensor-product heuristic.

# Function description
For supported rules, this estimator:
1) Builds the 1D quadrature nodes/weights for the given `rule`.
2) For each axis, integrates the `m`-th derivative along that axis while fixing
   the other two coordinates at quadrature nodes.
3) Sums the three axis-wise contributions.

This matches the original implementation exactly, including loop ordering and
floating-point accumulation order.

# Arguments
- `f`: 3D integrand callable `f(x, y, z)` (function, closure, or callable struct).
- `a`, `b`: Cube domain bounds.
- `N`: Number of subdivisions per axis.
- `rule`: Integration rule symbol.

# Returns
- A `Float64` error estimate. If `rule` is not recognized, returns `0.0`.
"""
function estimate_error_3d(f, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2

    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx

    # term X: ∬ ∂_x^m f(x̄,y,z) dy dz
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

    # term Y: ∬ ∂_y^m f(x,ȳ,z) dx dz
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

    # term Z: ∬ ∂_z^m f(x,y,z̄) dx dy
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

# ============================================================
# 4D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_4d(f, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate integration error for a 4D integral over a hypercube domain `[a,b]^4`
using a derivative-based tensor-product heuristic.

# Function description
For supported rules, this estimator:
1) Builds the 1D quadrature nodes/weights for the given `rule`.
2) For each axis, integrates the `m`-th derivative along that axis while fixing
   the other three coordinates at quadrature nodes.
3) Sums the four axis-wise contributions.

This matches the original implementation exactly, including loop ordering and
floating-point accumulation order.

# Arguments
- `f`: 4D integrand callable `f(x, y, z, t)` (function, closure, or callable struct).
- `a`, `b`: Hypercube domain bounds.
- `N`: Number of subdivisions per axis.
- `rule`: Integration rule symbol.

# Returns
- A `Float64` error estimate. If `rule` is not recognized, returns `0.0`.
"""
function estimate_error_4d(f, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2
    t̄ = (aa+bb)/2

    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    # term X: ∭ ∂_x^m f(x̄,y,z,t) dy dz dt
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

    # term Y: ∭ ∂_y^m f(x,ȳ,z,t) dx dz dt
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

    # term Z: ∭ ∂_z^m f(x,y,z̄,t) dx dy dt
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

    # term T: ∭ ∂_t^m f(x,y,z,t̄) dx dy dz
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

# ============================================================
# Unified public API
# ============================================================

"""
    estimate_error(f, a, b, N, dim, rule) -> Float64

Unified interface for estimating integration error in 1–4 dimensions.

# Function description
Dispatches to the corresponding dimension-specific estimator:
- `dim == 1` → `estimate_error_1d`
- `dim == 2` → `estimate_error_2d`
- `dim == 3` → `estimate_error_3d`
- `dim == 4` → `estimate_error_4d`

# Arguments
- `f`: Integrand function (expects `dim` positional arguments).
- `a`, `b`: Bounds for each dimension (interpreted as scalar bounds for a hypercube `[a,b]^dim`).
- `N`: Number of subdivisions per axis (subject to rule constraints in 1D; higher-D estimators reuse the same rule nodes/weights).
- `dim`: Number of dimensions (`Int`).
- `rule`: Integration rule symbol.

# Returns
- A `Float64` error estimate. If `dim` is outside 1–4 or `rule` is not recognized
  by the selected estimator, returns `0.0`.
"""
function estimate_error(f, a, b, N, dim, rule)
    if dim == 1
        return estimate_error_1d(f, a, b, N, rule)
    elseif dim == 2
        return estimate_error_2d(f, a, b, N, rule)
    elseif dim == 3
        return estimate_error_3d(f, a, b, N, rule)
    elseif dim == 4
        return estimate_error_4d(f, a, b, N, rule)
    else
        return 0.0  # TODO: higher-dim error estimators
    end
end

end  # module ErrorEstimator