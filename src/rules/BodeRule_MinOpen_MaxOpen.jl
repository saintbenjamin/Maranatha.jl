# src/rules/BodeRule_MinOpen_MaxOpen.jl

module BodeRule_MinOpen_MaxOpen

export bode_rule_min_open_max_open, bode_open_chain_error6

"""
    bode_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int) -> Float64

Globally-open (endpoint-free) composite Boole (a.k.a. Bode/Boole) rule.

Grid convention:
  h  = (b-a)/N
  xj = a + j*h,  j = 0,1,...,N

This rule approximates:
  ∫_{x0}^{xN} f(x) dx
using ONLY interior samples f(x1),...,f(x_{N-1}) (no f(x0), no f(xN)).

Construction (mathematically exact for polynomials up to degree 5):
- Start from the standard composite Boole rule (degree-5 exact).
- Eliminate endpoint samples f(x0), f(xN) by 5th-degree Lagrange extrapolation:
    f(x0) =  6 f(x1) - 15 f(x2) + 20 f(x3) - 15 f(x4) +  6 f(x5) - 1 f(x6)
    f(xN) =  6 f(x_{N-1}) - 15 f(x_{N-2}) + 20 f(x_{N-3})
           - 15 f(x_{N-4}) +  6 f(x_{N-5}) - 1 f(x_{N-6})

Constraints:
- N must be divisible by 4 (Boole panels are 4 subintervals wide).
- N must be ≥ 16 so that the left/right 6-point endpoint stencils do not overlap.
"""
function bode_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int)::Float64
    (N % 4 == 0) || error("Open composite Boole requires N divisible by 4, got N = $N")
    (N >= 16)    || error("Open composite Boole requires N ≥ 16 (non-overlapping 6-point end stencils), got N = $N")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    # Base composite Boole weights on interior nodes (closed rule):
    #   j mod 4 == 1 or 3 -> 64/45
    #   j mod 4 == 2      ->  8/15
    #   j mod 4 == 0      -> 28/45   (interior panel boundaries)
    # Endpoints (j=0,N) would be 14/45, but we do not sample them here.

    w_mod13 = 64.0 / 45.0
    w_mod2  =  8.0 / 15.0
    w_mod0  = 28.0 / 45.0

    # Endpoint elimination coefficients for degree-5 exact extrapolation:
    # f(x0) from x1..x6, and f(xN) from x_{N-1}..x_{N-6}.
    c1, c2, c3, c4, c5, c6 = 6.0, -15.0, 20.0, -15.0, 6.0, -1.0

    # Endpoint weights of the closed composite Boole rule (that we eliminate):
    w_end = 14.0 / 45.0

    # Precompute the modified left-end weights for j=1..6
    # w'_j = w_closed(j) + w_end * c_j
    function w_closed(j::Int)::Float64
        r = j % 4
        if r == 0
            return w_mod0
        elseif r == 2
            return w_mod2
        else
            return w_mod13
        end
    end

    wL = Vector{Float64}(undef, 6)
    wL[1] = w_closed(1) + w_end * c1
    wL[2] = w_closed(2) + w_end * c2
    wL[3] = w_closed(3) + w_end * c3
    wL[4] = w_closed(4) + w_end * c4
    wL[5] = w_closed(5) + w_end * c5
    wL[6] = w_closed(6) + w_end * c6

    # By symmetry (for N divisible by 4), the right-end modified weights match the left ones:
    # j = N-1..N-6 correspond to offsets 1..6 with the same c_k.
    wR = wL

    s = 0.0

    # Left end stencil (j = 1..6)
    @inbounds for j in 1:6
        s += wL[j] * f(aa + j*h)
    end

    # Middle region (j = 7..N-7) uses the unchanged closed-rule interior weights
    @inbounds for j in 7:(N-7)
        r = j % 4
        if r == 0
            s += w_mod0 * f(aa + j*h)
        elseif r == 2
            s += w_mod2 * f(aa + j*h)
        else
            s += w_mod13 * f(aa + j*h)
        end
    end

    # Right end stencil (j = N-6..N-1), mapped by offset k = N-j
    @inbounds for k in 6:-1:1
        j = N - k
        s += wR[k] * f(aa + j*h)
    end

    return h * s
end

"""
    bode_open_chain_error6(f::Function, a::Real, b::Real, N::Int; nth_derivative::Function) -> Float64

A practical leading-order error estimate based on a 6th-derivative model.

Important note:
- The quadrature `bode_rule_min_open_max_open` is degree-5 exact by construction.
- Exact higher-order remainder terms for this *endpoint-eliminated* open rule are more complicated
  than the textbook single-panel Boole remainder.
- This function provides a *heuristic* global estimate using f^(6) at the interval midpoint.

Model:
  error ≈ - C * (b-a) * h^6 * f^(6)( (a+b)/2 )

We set C = 8/945 to match the standard composite Boole leading constant in the closed case.
Use this as a sanity/scale indicator, not as a rigorous bound.
"""
function bode_open_chain_error6(
    f::Function, a::Real, b::Real, N::Int;
    nth_derivative::Function
)::Float64
    (N % 4 == 0) || error("Open composite Boole error6 requires N divisible by 4, got N = $N")
    (N >= 16)    || error("Open composite Boole error6 requires N ≥ 16, got N = $N")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xmid = (aa + bb) / 2
    d6   = nth_derivative(f, xmid, 6)

    C = 8.0 / 945.0
    return -(C * (bb - aa)) * h^6 * d6
end

end # module