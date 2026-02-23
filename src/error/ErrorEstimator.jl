module ErrorEstimator

using ForwardDiff
using LinearAlgebra
import ..BodeRule_MinOpen_MaxOpen

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
function estimate_error_2d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)
    h = (b - a) / N
    x = (a + b) / 2
    y = (a + b) / 2

    f2(v) = f(v[1], v[2])  # convert to vector-input form

    if rule == :simpson13_close || rule == :simpson38_close
        # Use Hessian to get 4th mixed partial derivative estimate
        H = ForwardDiff.hessian(f2, [x, y])
        est = H[1,1] + H[2,2]  # ∂²f/∂x² + ∂²f/∂y² (simplified proxy)
        return -(h^4 / 36) * est

    elseif rule == :bode_close
        H = ForwardDiff.hessian(f2, [x, y])
        # Approximate 6th derivative using the cube of 2nd derivatives (∂²f/∂x² and ∂²f/∂y²)
        # This is a heuristic proxy, not an exact sixth derivative
        est = H[1,1]^3 + H[2,2]^3
        return -(h^6 / 100.0) * est

    else
        return 0.0
    end
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
function estimate_error_3d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)
    h = (b - a) / N
    x = (a + b) / 2
    y = (a + b) / 2
    z = (a + b) / 2

    f3(v) = f(v[1], v[2], v[3])  # convert to vector-input form

    if rule == :simpson13_close || rule == :simpson38_close
        H = ForwardDiff.hessian(f3, [x, y, z])
        # Use trace of Hessian (sum of ∂²f/∂x², ∂²f/∂y², ∂²f/∂z²) as a second-derivative proxy
        return -(h^4 / 64) * tr(H)

    elseif rule == :bode_close
        H = ForwardDiff.hessian(f3, [x, y, z])
        # Approximate sixth derivative using cube of diagonal Hessian terms
        # Heuristic: ∂²f/∂x²³ + ∂²f/∂y²³ + ∂²f/∂z²³
        est = H[1,1]^3 + H[2,2]^3 + H[3,3]^3
        return -(h^6 / 1000.0) * est

    else
        return 0.0
    end
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
function estimate_error_4d(f::Function, a::Real, b::Real, N::Int, rule::Symbol)
    h = (b - a) / N
    x = (a + b) / 2
    y = (a + b) / 2
    z = (a + b) / 2
    t = (a + b) / 2

    f4(v) = f(v[1], v[2], v[3], v[4])  # convert to vector-input form

    if rule == :simpson13_close || rule == :simpson38_close
        H = ForwardDiff.hessian(f4, [x, y, z, t])
        # Use trace of Hessian (sum of ∂²f/∂x², ∂²f/∂y², ∂²f/∂z², ∂²f/∂t²)
        return -(h^4 / 100) * tr(H)

    elseif rule == :bode_close
        H = ForwardDiff.hessian(f4, [x, y, z, t])
        # Approximate sixth derivative using cube of diagonal Hessian terms
        est = H[1,1]^3 + H[2,2]^3 + H[3,3]^3 + H[4,4]^3
        return -(h^6 / 5000.0) * est

    else
        return 0.0
    end
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