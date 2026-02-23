# src/rules/Simpson38Rule_MinOpen_MaxOpen.jl

module Simpson38Rule_MinOpen_MaxOpen

export simpson38_rule_min_open_max_open, simpson38_rule_min_open_max_open_with_asymptotic_correction

"""
    simpson38_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int) -> Float64

Endpoint-free ("open") chained 3-point Newton–Cotes composite rule on a uniform grid.

This is the *true* endpoint-free open rule that tiles the interval by panels of width 4h:
  panel k integrates [x_{4k}, x_{4k+4}] using only interior nodes x_{4k+1}, x_{4k+2}, x_{4k+3}.

Grid:
  x_j = a + j*h,  h = (b-a)/N

Quadrature:
  ∫_{x0}^{xN} f(x) dx ≈ h * Σ_{k=0..M-1} [ (8/3) f(x_{4k+1}) - (4/3) f(x_{4k+2}) + (8/3) f(x_{4k+3}) ]
where N = 4M.

# Constraints
- N must be divisible by 4
- N must be at least 4
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


"""
    simpson38_rule_min_open_max_open_with_asymptotic_correction(f, a, b, N; nth_derivative) -> (Iquad, correction)

Returns:
- Iquad: quadrature value from `simpson38_rule_min_open_max_open`
- correction: asymptotic correction term (to be ADDED to Iquad) using derivatives at panel centers.

Asymptotic expansion per panel (center at x_{4k+2}):
  ∫ f - Q  =  + (14/45) h^5 f^(4)(x_{4k+2})
            + (41/945) h^7 f^(6)(x_{4k+2})
            + (61/22680) h^9 f^(8)(x_{4k+2})
            + O(h^11)

So the *correction to add to Q* is exactly that RHS summed over panels.

You must pass `nth_derivative(f, x, n)` as a callable keyword argument.
"""
function simpson38_rule_min_open_max_open_with_asymptotic_correction(
    f::Function, a::Real, b::Real, N::Int;
    nth_derivative::Function
)
    Iquad = simpson38_rule_min_open_max_open(f, a, b, N)

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N
    M  = N ÷ 4

    c = 0.0
    for k in 0:(M-1)
        xc = aa + (4k + 2)*h
        c += (14.0/45.0)    * h^5 * nth_derivative(f, xc, 4)
        c += (41.0/945.0)   * h^7 * nth_derivative(f, xc, 6)
        c += (61.0/22680.0) * h^9 * nth_derivative(f, xc, 8)
    end

    return Iquad, c
end

end # module