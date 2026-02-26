# ============================================================================
# src/integrands/F0000GammaEminus1.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module F0000GammaEminus1

# =========================
# F0000 integral via Maranatha (1D quadrature on y ∈ [0, 1])
# =========================

using SpecialFunctions

export gtilde_F0000

"""
    exI0_safe(
        x::T
    ) where {T<:Number}

Compute `exp(-x) * I0(x)` (modified Bessel I0) in an overflow-safe manner.

# Function description
This helper evaluates the scaled Bessel factor `exp(-x) * besseli(0, x)` that
appears in the F0000 integrand.

- For small to moderate `x`, it computes the expression directly.
- For large `x`, it switches to an asymptotic series for `exp(-x) I0(x)` to
  avoid overflow in `besseli(0, x)` while preserving the intended scaling.

# Arguments
- `x::T`: Input value (intended usage: `x ≥ 0`).

# Returns
- `T`: The value of `exp(-x) * I0(x)`.

# Notes
- The branch threshold `x ≤ 50` is a Float64-oriented "safe zone" heuristic and
  is kept identical to the original implementation for reproducibility.
- The asymptotic series used is:
  `exp(-x) I0(x) ≈ 1/sqrt(2πx) * (1 + 1/(8x) + 9/(128x^2) + 225/(3072x^3) + 11025/(98304x^4))`.
  This is a truncated large-`x` expansion and is not a strict error bound.
- The implementation is written to accept generic `Number` inputs so it can work
  with AD / Taylor objects when needed.

# Errors
- No explicit domain checks are performed. If `x` is negative, the asymptotic
  branch is not mathematically intended and may produce complex values due to
  the square root.
"""
@inline function exI0_safe(
    x::T
) where {T<:Number}
    if float(x) ≤ T(50) # if x ≤ T(50)  # threshold: safe zone for besseli(0,x) in Float64
        return exp(-x) * besseli(0, x)
    else
        invx = inv(x)
        # asymptotic series for exp(-x) I0(x)
        # 1/sqrt(2πx) * (1 + 1/(8x) + 9/(128x^2) + 225/(3072x^3) + 11025/(98304x^4))
        s = one(T) +
            invx / T(8) +
            T(9) * invx^2 / T(128) +
            T(225) * invx^3 / T(3072) +
            T(11025) * invx^4 / T(98304)
        return s / sqrt(T(2) * T(pi) * x)
    end
end

"""
    g_F0000_raw(
        y::T
    ) where {T<:Number}

Evaluate the raw integrand `g_F0000_raw(y)` used in the F0000 computation
for `y ∈ (0, 1)`.

# Function description
This function maps

`x = (1 - y) / y`  (so `x > 0` for `0 < y < 1`),

then forms the raw integrand as the sum of two pieces:

- `termA`: proportional to `(exp(-x) I0(x))^4` with a rational prefactor in `y`,
- `termB`: a correction term involving `exp(-x/2)` and a rational prefactor in `y`.

The algebraic form is preserved exactly from the original implementation.

# Arguments
- `y::T`: Real input (intended usage: `0 < y < 1`).

# Returns
- `T`: Raw integrand value at `y`.

# Notes
- This raw form contains endpoint-singular prefactors such as `1/y^3` and
  `1/(y*(1-y))`. It is therefore intended to be evaluated away from `y = 0, 1`.
  Endpoint suppression / regularization is handled in `gtilde_F0000` via the
  `t`-space cutoff.
- The implementation accepts generic `Number` to support AD / Taylor fallbacks.

# Errors
- No explicit domain checks are performed. Passing `y ≤ 0` or `y ≥ 1` can lead
  to division by zero or non-finite values.
"""
function g_F0000_raw(
    y::T
) where {T<:Number}
    x = (one(T) - y) / y  # x>0

    exI0 = exI0_safe(x)

    termA = (4T(pi)^2) * ((one(T) - y) / (y^3)) * (exI0^4)

    emx2 = exp(-x / T(2))
    bracket = one(T) - (T(1)/T(2)) * (one(T) + one(T)/y) * emx2
    termB = - (one(T) / (y * (one(T) - y))) * bracket

    return termA + termB
end

"""
    gtilde_F0000(
        t::T; 
        p::Int=2, 
        eps::T=T(1e-15)
    ) where {T<:Number}

Return the transformed integrand `g̃(t)` for the F0000 integral after the
variable substitution `y = t^p` on the interval `t ∈ [0, 1]`.

# Function description
This routine wraps `g_F0000_raw(y)` by applying:

- substitution: `y = t^p` with integer `p ≥ 1`,
- Jacobian: `dy = p * t^(p-1) dt`,

so the returned transformed integrand is:

`g̃(t) = p * t^(p-1) * g_F0000_raw(t^p)`.

To avoid endpoint singularities inherited from the raw `y`-integrand, the
function returns `0` when `t` is within `eps` of either endpoint.

# Arguments
- `t::T`: Parameter in `[0, 1]`.

# Keyword arguments
- `p::Int=2`: Power used in the substitution `y = t^p`.
- `eps::T=T(1e-15)`: Endpoint cutoff. If `t ≤ eps` or `1 - t ≤ eps`, returns `0`.

# Returns
- `T`: The transformed integrand value `g̃(t)` (or zero near endpoints).

# Notes
- Endpoint suppression is implemented exactly (hard cutoff) to keep numerical
  behavior unchanged and to avoid non-finite weights in quadrature rules that
  sample near `t = 0` or `t = 1`.
- The function is generic in `T<:Number` so it can be used with AD / Taylor
  objects, but the cutoff comparisons (`t ≤ eps`) require an ordered type; this
  is intended for real-valued `t`.

# Errors
- Throws an error if `p < 1` (implicitly, via `t^(p-1)` or user intent); no
  explicit check is performed here to preserve original behavior.
"""
function gtilde_F0000(
    t::T; 
    p::Int=2, 
    eps::T=T(1e-15)
) where {T<:Number}
    if t ≤ eps
        return zero(T)
    elseif (one(T) - t) ≤ eps
        return zero(T)
    end

    y = t^p
    return T(p) * t^(p-1) * g_F0000_raw(y)
    # return g_F0000_raw(t)
end

end  # module F0000GammaEminus1