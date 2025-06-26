# src/rules/Simpson13Rule.jl

module Simpson13Rule

export simpson13_rule

"""
    simpson13_rule(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using Simpson’s 1/3 rule.

# Arguments
- `f`: Function of one variable, `f(x)`
- `a`, `b`: Integration limits
- `N`: Number of subintervals (must be even)

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]`

# Notes
- Simpson’s 1/3 rule is a second-order Newton–Cotes method
- The interval `[a, b]` is divided into `N` equal subintervals of width `h = (b - a)/N`
- Composite rule: uses weights `[1, 4, 2, 4, ..., 2, 4, 1]`
"""
function simpson13_rule(f::Function, a::Float64, b::Real, N::Int)
    if N % 2 != 0
        error("Simpson's 1/3 rule requires N divisible by 2, got N = $N")
    end

    h = (b - a) / N
    x = [a + i * h for i in 0:N]
    result = f(x[1]) + f(x[end])

    for i in 2:2:N
        result += 4 * f(x[i])
    end
    for i in 3:2:N-1
        result += 2 * f(x[i])
    end

    return (h / 3) * result
end

end # module