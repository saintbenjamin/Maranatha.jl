module Simpson38Rule

export simpson38_rule

"""
    simpson38_rule_1d(f::Function, a::Float64, b::Real, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using Simpson’s 3/8 rule  
(composite Newton–Cotes with 4-point stencil).  

# Arguments
- `f`: Function of one variable, `f(x)`
- `a`, `b`: Integration limits
- `N`: Number of subintervals (must be divisible by 3)

# Returns
- `Float64`: Estimated definite integral of `f(x)` over `[a, b]`

# Notes
- Simpson’s 3/8 rule uses weights: `[1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1]`
- The method requires `N ≡ 0 mod 3`, as it applies the 4-point rule repeatedly
"""
function simpson38_rule(f::Function, a::Float64, b::Real, N::Int)
    if N % 3 != 0
        error("Simpson 3/8 rule requires N divisible by 3, got N = $N")
    end

    h = (b - a) / N
    x = [a + i * h for i in 0:N]
    result = f(x[1]) + f(x[end])

    for i in 2:3:N
        result += 3 * f(x[i])
    end
    for i in 3:3:N-1
        result += 2 * f(x[i])
    end
    for i in 4:3:N-2
        result += 3 * f(x[i])
    end

    return (3h / 8) * result
end

end # module