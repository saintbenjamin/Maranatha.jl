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
function simpson13_rule(f, a, b, N)
    @assert N % 2 == 0 "N must be even for Simpson's 1/3 rule"
    h = (b - a) / N
    x = a:h:b
    y = f.(x)
    return h/3 * (y[1] + y[end] + 4*sum(y[2:2:end-1]) + 2*sum(y[3:2:end-2]))
end

end # module