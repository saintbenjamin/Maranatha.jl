module ErrorEstimator

using ForwardDiff
using LinearAlgebra
using ..Integrate
using ..BodeRule_MinOpen_MaxOpen

export estimate_error

# n-th derivative of f at x (scalar) via repeated ForwardDiff.derivative
function nth_derivative(f::Function, x::Real, n::Int)
    g = f
    for _ in 1:n
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
    end
    return g(x)
end

"""
    estimate_error_1d(f::Function, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate the integration error for a 1D integral of `f(x)` over `[a, b]`  
using Newton–Cotes quadrature rules and automatic differentiation.

This function uses centered higher-order derivatives (4th–10th) to  
approximate the leading error terms for each rule, based on symbolic  
error expansions.

# Arguments
- `f`: Integrand function `f(x)`
- `a`, `b`: Integration limits
- `N`: Number of subintervals (must be divisible by rule-specific block size)
- `rule`: Integration rule symbol:
    - `:simpson13_close` → Simpson’s 1/3 rule (4th, 6th, 8th derivative)
    - `:simpson38_close` → Simpson’s 3/8 rule (4th, 6th, 8th derivative)
    - `:bode_close`      → Bode’s rule (6th, 8th, 10th derivative)

# Returns
- Approximate total integration error as a sum of leading derivative-based terms.
"""
function estimate_error_1d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)
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

        err = BodeRule_MinOpen_MaxOpen.bode_open_chain_error6(f, aa, bb, N; nth_derivative=nth_derivative)

        return err

    else
        return 0.0
    end
end

"""
    estimate_error_2d(f::Function, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate integration error for a 2D integral using the trace of the Hessian  
(for Simpson rules) or an approximate 6th derivative (for Bode rule).

# Arguments
- `f`: 2D integrand `f(x, y)`
- `a`, `b`: Square domain bounds
- `N`: Number of subdivisions per axis
- `rule`: Integration rule symbol

# Returns
- Estimated numerical integration error based on local curvature or higher derivatives
"""
# ============================================================
# 2D derivative-based tensor-product error
# ============================================================
function estimate_error_2d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2

    # map rule → derivative order and constant
    if rule == :simpson13_close
        m = 4; C = -1/180
    elseif rule == :simpson38_close
        m = 4; C = -1/80
    elseif rule == :bode_close
        m = 6; C = -2/945
    elseif rule == :simpson13_open
        m = 3; C = -3/8
    elseif rule == :simpson38_open
        m = 4; C = 14/45
    elseif rule == :bode_open
        m = 6; C = 1.0
    else
        return 0.0
    end

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

"""
    estimate_error_3d(f::Function, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate integration error for a 3D integrand over a cube domain using  
2nd-order Hessian trace (for Simpson) or nested 6th derivative (for Bode).

# Arguments
- `f`: 3D integrand `f(x, y, z)`
- `a`, `b`: Cube bounds
- `N`: Grid resolution
- `rule`: Integration rule symbol

# Returns
- Numerical estimate of leading-order integration error
"""
# ============================================================
# 3D derivative-based tensor-product error (leading terms)
# ============================================================
function estimate_error_3d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2

    # rule → derivative order m and constant C
    if rule == :simpson13_close
        m = 4; C = -1/180
    elseif rule == :simpson38_close
        m = 4; C = -1/80
    elseif rule == :bode_close
        m = 6; C = -2/945
    elseif rule == :simpson13_open
        m = 3; C = -3/8
    elseif rule == :simpson38_open
        m = 4; C = 14/45
    elseif rule == :bode_open
        m = 6; C = 1.0
    else
        return 0.0
    end

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

"""
    estimate_error_4d(f::Function, a::Real, b::Real, N::Int, rule::Symbol) -> Float64

Estimate error for 4D numerical integration using trace of Hessian  
or approximated 6th derivative with ForwardDiff.

# Arguments
- `f`: 4D integrand `f(x, y, z, t)`
- `a`, `b`: Hypercube bounds
- `N`: Resolution per axis
- `rule`: Integration rule symbol

# Returns
- Estimated quadrature error using local curvature or higher-order terms
"""
# ============================================================
# 4D derivative-based tensor-product error (leading terms)
# ============================================================
function estimate_error_4d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2
    t̄ = (aa+bb)/2

    # rule → derivative order m and constant C
    if rule == :simpson13_close
        m = 4; C = -1/180
    elseif rule == :simpson38_close
        m = 4; C = -1/80
    elseif rule == :bode_close
        m = 6; C = -2/945
    elseif rule == :simpson13_open
        m = 3; C = -3/8
    elseif rule == :simpson38_open
        m = 4; C = 14/45
    elseif rule == :bode_open
        m = 6; C = 1.0
    else
        return 0.0
    end

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

"""
    estimate_error(f, a, b, N, dim, rule) -> Float64

Unified interface for estimating integration error in 1–4 dimensions.

# Arguments
- `f`: Integrand function (1–4 variables)
- `a`, `b`: Bounds for each dimension
- `N`: Number of subdivisions
- `dim`: Number of dimensions
- `rule`: Integration rule symbol

# Returns
- Approximate error based on rule-dependent derivative estimates
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

end  # module