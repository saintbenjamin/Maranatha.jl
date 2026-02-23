# ============================================================================
# src/rules/Simpson38Rule_MinOpen_MaxOpen.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Simpson38Rule_MinOpen_MaxOpen

export simpson38_rule_min_open_max_open

"""
    simpson38_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using an endpoint-free
("open") chained 3-point Newton–Cotes composite rule on a uniform grid.

# Function description
This is a truly endpoint-free open Newton–Cotes rule that tiles the full interval
by panels of width `4h`. Each panel `k` approximates the integral on
`[x_{4k}, x_{4k+4}]` using only the interior nodes
`x_{4k+1}, x_{4k+2}, x_{4k+3}` (no endpoint evaluations on panel boundaries).

The uniform grid is defined by:
- `h = (b - a)/N`
- `x_j = a + j*h`, for `j = 0,1,...,N`.

The quadrature is:
```
∫*{x0}^{xN} f(x) dx ≈ h * Σ*{k=0..M-1} [
(8/3) f(x_{4k+1}) - (4/3) f(x_{4k+2}) + (8/3) f(x_{4k+3})
]
```
with `N = 4M`.

The implementation preserves the original evaluation order and arithmetic.

# Arguments
- `f`: Integrand function of one variable, `f(x)`.
- `a::Real`: Lower integration bound.
- `b::Real`: Upper integration bound.
- `N::Int`: Number of subintervals (must satisfy the constraints below).

# Returns
- Estimated value of the definite integral over `[a, b]` as a `Float64`.

# Constraints
- `N` must be divisible by 4.
- `N ≥ 4`.

# Notes
- The rule is endpoint-free: it does not evaluate `f(a)` or `f(b)`, and also does
  not evaluate the panel boundary nodes `x_{4k}` and `x_{4k+4}`.
- The panel width is exactly `4h`, so the number of panels is `M = N ÷ 4`.

# Errors
- Throws an error if `N` is not divisible by 4 or if `N < 4`.
"""
function simpson38_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int)::Float64
    if N % 4 != 0
        error("Open 3-point chained rule requires N divisible by 4 (panel width = 4h), got N = $N")
    end
    if N < 4
        error("Open 3-point chained rule requires N ≥ 4, got N = $N")
    end

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    M = N ÷ 4
    s = 0.0

    for k in 0:(M-1)
        j1 = 4k + 1
        j2 = 4k + 2
        j3 = 4k + 3
        s += (8.0/3.0)  * f(aa + j1*h)
        s += (-4.0/3.0) * f(aa + j2*h)
        s += (8.0/3.0)  * f(aa + j3*h)
    end

    return h * s
end

end  # module Simpson38Rule_MinOpen_MaxOpen