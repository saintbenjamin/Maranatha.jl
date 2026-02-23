# ============================================================================
# src/rules/BodeRule_MinOpen_MaxOpen.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BodeRule_MinOpen_MaxOpen

export bode_rule_min_open_max_open, bode_rule_min_open_max_open_error

"""
    bode_rule_min_open_max_open(f::Function, a::Real, b::Real, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using a globally-open
(endpoint-free) composite Boole rule (a.k.a. Bode/Boole) on a uniform grid.

# Function description
This rule approximates the definite integral on `[a, b]` using only interior
samples `f(x1), ..., f(x_{N-1})`, i.e. it never evaluates the integrand at the
endpoints `x0 = a` or `xN = b`.

Grid convention:
- `h  = (b - a)/N`
- `xj = a + j*h`, for `j = 0,1,...,N`.

Construction summary (degree-5 exactness):
- Start from the standard composite closed Boole rule, which is exact for
  polynomials up to degree 5.
- Eliminate the endpoint samples `f(x0)` and `f(xN)` using 5th-degree Lagrange
  extrapolation expressed in terms of the first/last six interior nodes:
  - `f(x0)` from `x1..x6`
  - `f(xN)` from `x_{N-1}..x_{N-6}`

The resulting endpoint-eliminated rule remains degree-5 exact while being
globally endpoint-free. The implementation preserves the original evaluation
order and arithmetic.

# Arguments
- `f`: Integrand function of one variable, `f(x)`.
- `a::Real`: Lower integration bound.
- `b::Real`: Upper integration bound.
- `N::Int`: Number of subintervals (must satisfy the constraints below).

# Returns
- Estimated value of the definite integral over `[a, b]` as a `Float64`.

# Constraints
- `N` must be divisible by 4 (Boole panels are 4 subintervals wide).
- `N ≥ 16` so that the left/right 6-point endpoint stencils do not overlap.

# Notes
- Base interior weights (from the closed composite Boole rule) depend on `j mod 4`:
  - `j mod 4 == 1 or 3` → `64/45`
  - `j mod 4 == 2`      → `8/15`
  - `j mod 4 == 0`      → `28/45`  (interior panel boundaries)
  - endpoints `j=0,N` would be `14/45`, but they are not sampled here.
- Endpoint elimination uses coefficients `(6, -15, 20, -15, 6, -1)` applied to
  the first/last six interior nodes.

# Errors
- Throws an error if `N` is not divisible by 4 or if `N < 16`.
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
    bode_rule_min_open_max_open_error(f::Function, a::Real, b::Real, N::Int; nth_derivative::Function) -> Float64

Estimate the integration error scale for `bode_rule_min_open_max_open` using a
6th-derivative leading-order model.

# Function description
The endpoint-eliminated open Boole rule is degree-5 exact by construction, but
its exact higher-order remainder terms are more complicated than the textbook
single-panel Boole remainder. This function therefore provides a practical,
heuristic global error estimate based on the 6th derivative of `f` evaluated at
the interval midpoint.

Model:
`error ≈ -C * (b - a) * h^6 * f^(6)((a+b)/2)`

The constant is set to `C = 8/945`, matching the standard composite Boole
leading constant in the closed case. This is intended as a scale/sanity
indicator rather than a rigorous bound.

# Arguments
- `f`: Integrand function of one variable, `f(x)`.
- `a::Real`: Lower integration bound.
- `b::Real`: Upper integration bound.
- `N::Int`: Number of subintervals (must satisfy the constraints below).

# Keyword arguments
- `nth_derivative::Function`: A function that computes the `n`-th derivative of
  `f` at a point, called as `nth_derivative(f, x, n)`.

# Returns
- Estimated error (as `exact - quadrature`) as a `Float64` under the model above.

# Constraints
- `N` must be divisible by 4.
- `N ≥ 16` (consistent with the open-chain rule constraints).

# Errors
- Throws an error if `N` violates the constraints.
"""
function bode_rule_min_open_max_open_error(
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

end  # module BodeRule_MinOpen_MaxOpen