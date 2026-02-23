# src/rules/Simpson13Rule_MinOpen_MaxOpen.jl

module Simpson13Rule_MinOpen_MaxOpen

export simpson13_rule_min_open_max_open

"""
    simpson13_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int) -> Float64

Globally-open (endpoint-free) composite Simpson 1/3 "open-chain" rule.

Grid:
  h  = (b-a)/N
  xj = a + j*h, j = 0,1,...,N

This rule approximates ∫_{x0}^{xN} f(x) dx using ONLY interior samples
f(x1),...,f(x_{N-1}) (no f(x0), no f(xN)).

Quadrature (degree-3 exact):
  ∫ f ≈ h * [
      (9/4)   f(x1)   + (13/12) f(x3)
    + (4/3)   Σ f(xj) for even j = 4,6,...,N-4
    + (2/3)   Σ f(xj) for odd  j = 5,7,...,N-5
    + (13/12) f(x_{N-3}) + (9/4) f(x_{N-1})
  ]

Constraints:
- N must be even
- N must be ≥ 8  (smallest N where the pattern is well-defined)
"""
function simpson13_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int)::Float64
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

end # module