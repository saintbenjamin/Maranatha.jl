# ============================================================================
# src/rules/Simpson13Rule_MinOpen_MaxOpen.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Simpson13Rule_MinOpen_MaxOpen

export simpson13_rule_min_open_max_open

"""
    simpson13_rule_min_open_max_open(
        f, 
        a::Real, 
        b::Real, 
        N::Int
    ) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using a globally-open
(endpoint-free) composite Simpson 1/3 "open-chain" rule.

# Function description
This rule approximates the definite integral over `[a, b]` using only interior
grid samples and never evaluates the integrand at the endpoints `x0 = a` or
`xN = b`. The grid is defined by

- `h  = (b - a)/N`
- `xj = a + j*h`, for `j = 0,1,...,N`.

The quadrature is exact for polynomials up to degree 3 and uses the stencil

```
∫ f(x) dx ≈ h * [
(9/4)   f(x1)   + (13/12) f(x3)

* (4/3)   Σ f(xj) for even j = 4,6,...,N-4
* (2/3)   Σ f(xj) for odd  j = 5,7,...,N-5
* (13/12) f(x_{N-3}) + (9/4) f(x_{N-1})
  ]
```

The implementation preserves the original evaluation order and arithmetic.

# Arguments
- `f`: Integrand callable of one variable `f(x)` (function, closure, or callable struct).
- `a::Real`: Lower integration bound.
- `b::Real`: Upper integration bound.
- `N::Int`: Number of subintervals (must satisfy the constraints below).

# Returns
- Estimated value of the definite integral over `[a, b]` as a `Float64`.

# Constraints
- `N` must be even.
- `N ≥ 8` (smallest size where the open-chain pattern is well-defined).

# Notes
- Endpoint values `f(a)` and `f(b)` are never evaluated.
- The rule uses asymmetric weights near the boundaries to maintain global
  degree-3 exactness while remaining endpoint-free.

# Errors
- Throws an error if `N` is not even or if `N < 8`.
"""
function simpson13_rule_min_open_max_open(
    f, 
    a::Real, 
    b::Real, 
    N::Int
)::Float64
    (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
    (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    s = 0.0

    # Boundary-near interior nodes (endpoints x0, xN are not sampled).
    s += (9.0  / 4.0)  * f(aa + 1.0 * h)          # x1
    s += (13.0 / 12.0) * f(aa + 3.0 * h)          # x3
    s += (13.0 / 12.0) * f(aa + (N - 3) * h)      # x_{N-3}
    s += (9.0  / 4.0)  * f(aa + (N - 1) * h)      # x_{N-1}

    # Even indices: 4,6,...,N-4  with weight 4/3.
    @inbounds for j in 4:2:(N - 4)
        s += (4.0 / 3.0) * f(aa + j * h)
    end

    # Odd indices: 5,7,...,N-5 with weight 2/3 (excluding 3 and N-3 handled above).
    @inbounds for j in 5:2:(N - 5)
        s += (2.0 / 3.0) * f(aa + j * h)
    end

    return h * s
end

end  # module Simpson13Rule_MinOpen_MaxOpen