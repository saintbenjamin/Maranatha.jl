# src/rules/BodeRule.jl

module BodeRule

export bode_rule

"""
    bode_rule(f::Function, a::Real, b::Real, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using the composite
Bode’s rule (closed Newton–Cotes on 5 points, i.e., 4 subintervals per block).

# Arguments
- `f`: Function of a single variable, `f(x)`
- `a`, `b`: Integration bounds
- `N`: Number of subintervals (must be divisible by 4)

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]`

# Notes
- Requires `N % 4 == 0`
- Per 4-subinterval block (5 points): weights `[7, 32, 12, 32, 7]`
- Step size: `h = (b - a) / N`
- Final scaling: `(2h/45) * Σ block_sum`
"""
function bode_rule(f::Function, a::Real, b::Real, N::Int)::Float64
    if N % 4 != 0
        error("Bode's rule requires N divisible by 4, got N = $N")
    end

    aa = float(a)
    bb = float(b)

    h = (bb - aa) / N
    total = 0.0

    nblocks = N ÷ 4
    for k in 0:(nblocks - 1)
        x0 = aa + (4 * k) * h
        x1 = x0 + h
        x2 = x0 + 2.0 * h
        x3 = x0 + 3.0 * h
        x4 = x0 + 4.0 * h

        total += 7.0 * f(x0) + 32.0 * f(x1) + 12.0 * f(x2) + 32.0 * f(x3) + 7.0 * f(x4)
    end

    return (2.0 * h / 45.0) * total
end

end # module BodeRule