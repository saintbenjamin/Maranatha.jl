# ============================================================================
# src/rules/BodeRule.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BodeRule

using ..JobLoggerTools

export bode_rule

"""
    bode_rule(
        f, 
        a::Real, 
        b::Real, 
        N::Int
    ) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using the composite
Bode’s rule (closed 5-point Newton–Cotes; 4 subintervals per block).

# Function description
This function applies the composite Bode’s rule by partitioning the interval
`[a, b]` into `N` uniform subintervals with spacing `h = (b-a)/N`, where `N` must
be divisible by 4. The integral is computed as a sum over `N/4` non-overlapping
blocks, each spanning 4 subintervals (5 grid points) with weights:

`[7, 32, 12, 32, 7]`.

The implementation preserves the original evaluation order and arithmetic.

# Arguments
- `f`: Integrand callable of one variable `f(x)` (function, closure, or callable struct).
- `a::Real`: Lower integration limit.
- `b::Real`: Upper integration limit.
- `N::Int`: Number of subintervals (must be divisible by 4).

# Returns
- Estimated value of the definite integral of `f(x)` over `[a, b]` as a `Float64`.

# Notes
- Step size: `h = (b - a) / N`.
- Number of blocks: `nblocks = N ÷ 4`.
- Each block contributes:
  `7 f(x0) + 32 f(x1) + 12 f(x2) + 32 f(x3) + 7 f(x4)`,
  with `xj = a + (4k + j) h` for `j = 0..4`.
- Composite scaling factor: `(2h/45)`.

# Errors
- Throws an error if `N` is not divisible by 4.
"""
function bode_rule(
    f, 
    a::Real, 
    b::Real, 
    N::Int
)::Float64
    if N % 4 != 0
        JobLoggerTools.error_benji("Close composite Boole's rule requires N divisible by 4, got N = $N")
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

end  # module BodeRule