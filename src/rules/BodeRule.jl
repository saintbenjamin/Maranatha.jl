module BodeRule

export bode_rule

# Bode's Rule (closed Newton-Cotes, 4 points)

"""
    bode_rule(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using Bode's rule  
(4-point closed Newton–Cotes). The number of intervals `N` must be divisible by 4.

# Arguments
- `f`: Function of a single variable, `f(x)`
- `a`, `b`: Integration bounds
- `N`: Number of subdivisions (must satisfy `N % 4 == 0`)

# Returns
- Estimated integral of `f` over `[a, b]` using Bode's rule
"""
function bode_rule(f::Function, a::Float64, b::Real, N::Int)
    if N % 4 != 0
        error("Bode's rule requires N divisible by 4")
    end

    h = (b - a) / N
    sum = 0.0

    for i in 0:(N ÷ 4 - 1)
        x0 = a + 4i * h
        x1 = x0 + h
        x2 = x1 + h
        x3 = x2 + h
        x4 = x3 + h

        sum += 7*f(x0) + 32*f(x1) + 12*f(x2) + 32*f(x3) + 7*f(x4)
    end

    return (2h / 45) * sum
end

end  # module BodeRule