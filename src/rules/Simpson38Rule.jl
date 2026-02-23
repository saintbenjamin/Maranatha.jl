# src/rules/Simpson38Rule.jl

module Simpson38Rule

export simpson38_rule

"""
    simpson38_rule(f::Function, a::Real, b::Real, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using the composite
Simpson’s 3/8 rule (closed Newton–Cotes, repeated 4-point stencil).

# Arguments
- `f`: Function of one variable, `f(x)`
- `a`, `b`: Integration limits
- `N`: Number of subintervals (must be divisible by 3)

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]`

# Notes
- Requires `N % 3 == 0`
- Composite weights:
  - endpoints: 1
  - interior points: weight 2 if the point index is a multiple of 3, else 3
- Step size: `h = (b - a) / N`
- Final scaling: `(3h/8) * Σ w_i f(x_i)`
"""
function simpson38_rule(f::Function, a::Real, b::Real, N::Int)::Float64
    if N % 3 != 0
        error("Simpson 3/8 rule requires N divisible by 3, got N = $N")
    end

    aa = float(a)
    bb = float(b)

    h = (bb - aa) / N

    # endpoints
    result = f(aa) + f(bb)

    # interior points: i = 1,2,...,N-1 (where x = a + i*h)
    # weight = 2 if i % 3 == 0 else 3
    for i in 1:(N - 1)
        xi = aa + i * h
        w  = (i % 3 == 0) ? 2.0 : 3.0
        result += w * f(xi)
    end

    return (3.0 * h / 8.0) * result
end

end # module Simpson38Rule