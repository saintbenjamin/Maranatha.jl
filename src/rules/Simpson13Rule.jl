# ============================================================================
# src/rules/Simpson13Rule.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Simpson13Rule

export simpson13_rule

"""
    simpson13_rule(
        f, 
        a::Float64, 
        b::Real, 
        N::Int
    ) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using the composite
Simpson’s 1/3 rule.

# Function description
This function applies the composite Simpson 1/3 Newton–Cotes quadrature rule on
a uniform grid of `N` subintervals. The endpoint contributions are weighted by
1, odd interior points are weighted by 4, and even interior points are weighted
by 2:

`[1, 4, 2, 4, ..., 2, 4, 1]`.

The implementation preserves the original evaluation order and arithmetic.

# Arguments
- `f`: Integrand callable of one variable `f(x)` (function, closure, or callable struct).
- `a::Float64`: Lower integration limit.
- `b::Real`: Upper integration limit.
- `N::Int`: Number of subintervals (must be divisible by 2).

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]` as a `Float64`.

# Notes
- The grid spacing is `h = (b - a)/N`.
- The grid nodes are constructed as `x[i] = a + (i-1)*h` for `i = 1..N+1`.
- This is a 4th-order accurate method for sufficiently smooth integrands under
  the standard composite Simpson assumptions.

# Errors
- Throws an error if `N` is not divisible by 2.
"""
function simpson13_rule(
    f, 
    a::Float64, 
    b::Real, 
    N::Int
)
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

end  # module Simpson13Rule