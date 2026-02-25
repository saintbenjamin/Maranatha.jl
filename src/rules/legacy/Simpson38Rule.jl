# ============================================================================
# src/rules/Simpson38Rule.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Simpson38Rule

export simpson38_rule

"""
    simpson38_rule(
        f, 
        a::Real, 
        b::Real, 
        N::Int
    ) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using the composite
Simpson’s 3/8 rule (closed Newton–Cotes with a repeated 4-point stencil).

# Function description
This function applies the composite Simpson 3/8 rule on a uniform grid of `N`
subintervals (so the node spacing is `h = (b-a)/N`). The rule requires `N` to be
divisible by 3 and uses weights:
- endpoints: weight `1`,
- interior points: weight `2` if the index is a multiple of 3, otherwise `3`.

The implementation preserves the original evaluation order and arithmetic.

# Arguments
- `f`: Integrand callable of one variable `f(x)` (function, closure, or callable struct).
- `a::Real`: Lower integration limit.
- `b::Real`: Upper integration limit.
- `N::Int`: Number of subintervals (must be divisible by 3).

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]` as a `Float64`.

# Notes
- The step size is `h = (b - a) / N`.
- The composite scaling factor is `(3h/8)`.
- This method is 4th-order accurate for sufficiently smooth integrands under the
  standard composite Simpson assumptions.

# Errors
- Throws an error if `N` is not divisible by 3.
"""
function simpson38_rule(
    f, 
    a::Real, 
    b::Real, 
    N::Int
)::Float64
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

end  # module Simpson38Rule