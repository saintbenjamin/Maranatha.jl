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
    exI0_safe(x::T) where {T<:Real}

Compute `exp(-x) * I0(x)` (modified Bessel I0) in an overflow-safe manner.

# Function description
For small to moderate `x`, this function evaluates `exp(-x) * besseli(0, x)`
directly. For large `x`, it uses an asymptotic expansion for `exp(-x) I0(x)`
to avoid overflow in `besseli(0, x)` while preserving the intended numerical
behavior.

# Arguments
- `x::T`: Nonnegative real input (intended usage: `x > 0`).

# Returns
- `T`: The value of `exp(-x) * I0(x)` computed using either direct evaluation
  or an asymptotic series (depending on `x`).

# Notes
- The threshold `x ≤ 50` is chosen as a "safe zone" heuristic for Float64
  evaluation of `besseli(0, x)` without overflow. The threshold is applied in
  type `T` and kept identical to the original implementation.
- The asymptotic series used is:
  `exp(-x) I0(x) ≈ 1/sqrt(2πx) * (1 + 1/(8x) + 9/(128x^2) + 225/(3072x^3) + 11025/(98304x^4))`.
"""
@inline function exI0_safe(x::T) where {T<:Real}
    if x ≤ T(50)  # threshold: safe zone for besseli(0,x) in Float64
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
    g_F0000_raw(y::T) where {T<:Real}

Evaluate the raw integrand `g_F0000_raw(y)` used in the F0000 computation
for `y ∈ (0, 1)`.

# Function description
This function maps `y` to `x = (1-y)/y` (with `x > 0` for `y ∈ (0,1)`),
evaluates the overflow-safe `exp(-x) I0(x)` factor, and combines two terms:
- `termA`: proportional to `(exp(-x) I0(x))^4` with a rational prefactor in `y`,
- `termB`: a correction term involving `exp(-x/2)` and a rational prefactor in `y`.

The expression is preserved exactly from the original implementation.

# Arguments
- `y::T`: Real input (intended usage: `0 < y < 1`).

# Returns
- `T`: The raw integrand value at `y`.

# Notes
- The formula contains factors like `1/y^3` and `1/(y*(1-y))`, so it is intended
  for use away from the endpoints. Endpoint handling is implemented in
  `gtilde_F0000`.
"""
function g_F0000_raw(y::T) where {T<:Real}
    x = (one(T) - y) / y  # x>0

    exI0 = exI0_safe(x)

    termA = (4T(pi)^2) * ((one(T) - y) / (y^3)) * (exI0^4)

    emx2 = exp(-x / T(2))
    bracket = one(T) - (T(1)/T(2)) * (one(T) + one(T)/y) * emx2
    termB = - (one(T) / (y * (one(T) - y))) * bracket

    return termA + termB
end

"""
    gtilde_F0000(t::T; p::Int=2, eps::T=T(1e-15)) where {T<:Real}

Return the transformed integrand `g̃(t)` for the F0000 integral after the
variable substitution `y = t^p` on the interval `t ∈ [0, 1]`.

# Function description
This function applies the substitution `y = t^p` (with integer `p ≥ 1`) and
returns the Jacobian-weighted integrand:

`g̃(t) = p * t^(p-1) * g_F0000_raw(t^p)`.

To avoid singular behavior at the endpoints, it returns zero when `t` is within
`eps` of `0` or `1`.

# Arguments
- `t::T`: Parameter in `[0, 1]`.

# Keyword arguments
- `p::Int=2`: Power used in the substitution `y = t^p`.
- `eps::T=T(1e-15)`: Endpoint cutoff. If `t ≤ eps` or `1 - t ≤ eps`, the function
  returns zero.

# Returns
- `T`: The transformed integrand value `g̃(t)` (or zero near the endpoints).

# Notes
- Endpoint suppression is implemented exactly as in the original code to keep
  numerical behavior unchanged.
"""
function gtilde_F0000(t::T; p::Int=2, eps::T=T(1e-15)) where {T<:Real}
    if t ≤ eps
        return zero(T)
    elseif (one(T) - t) ≤ eps
        return zero(T)
    end

    y = t^p
    return T(p) * t^(p-1) * g_F0000_raw(y)
end

end  # module F0000GammaEminus1