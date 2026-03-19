# ============================================================================
# test/experiments/F0000GammaEminus1.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# =========================
# F0000 integral via Maranatha (1D quadrature on y ∈ [0, 1])
# =========================

using SpecialFunctions
using Bessels

# ============================================================
# TaylorSeries compatibility shims (local to this module)
# - Provide besseli(0, Taylor1) via local Taylor expansion
# - Make branch comparisons Taylor-safe by using constant terms
# ============================================================

using TaylorSeries
using ForwardDiff
using FastDifferentiation

import SpecialFunctions: besseli

"""
    _const_term(x)

Return the constant term of `x`.

# Function description
If `x` is a `TaylorSeries.Taylor1`, this function extracts the constant
coefficient (`x.coeffs[1]`). Otherwise it returns `x` unchanged.

# Arguments
- `x`: Any numeric value or `Taylor1`.

# Returns
- The constant term of `x`.

# Notes
- Used to evaluate branch conditions using the real constant part of Taylor
  inputs while preserving the original object for subsequent computation.
"""
@inline _const_term(x) = x
@inline _const_term(x::TaylorSeries.Taylor1) = x.coeffs[1]

"""
    _nth_derivative_forwarddiff_scalar(g, x::Real, n::Int)

Compute the `n`-th derivative of a scalar function `g` at point `x`
using repeated `ForwardDiff.derivative`.

# Function description
This helper constructs the `n`-th derivative operator by repeatedly
wrapping `ForwardDiff.derivative`. It is primarily used to obtain the
Taylor coefficients of special functions that do not natively support
`Taylor1` inputs.

# Arguments
- `g`: Scalar function `g(::Real)`.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order (`n ≥ 0`).

# Returns
- `Real`: The value of `g^{(n)}(x)`.

# Notes
- Intended for small derivative orders (as required by local Taylor
  expansions in error estimation).
"""
@inline function _nth_derivative_forwarddiff_scalar(g, x::Real, n::Int)
    n >= 0 || throw(ArgumentError("n must be ≥ 0 (got n=$n)"))
    h = g
    for _ in 1:n
        prev = h
        h = t -> ForwardDiff.derivative(prev, t)
    end
    return h(x)
end

"""
    besseli(nu::Integer, x::TaylorSeries.Taylor1{T}) where {T<:Real}

Provide `TaylorSeries` support for `besseli(0, x)`.

# Function description
`SpecialFunctions.besseli` does not define a method for `Taylor1` inputs.
This method constructs the Taylor expansion of `I₀(x)` around the constant
term `x₀`:

```math
I_0(x_0 + dx) = \\sum_{k=0}^{N} \\frac{I_0^k(x_0)}{k!} dx^k
```
where ``dx = x - x_0`` and `N = x.order`.

The derivatives ``I₀^k(x_0)`` are evaluated using repeated
`ForwardDiff.derivative`.

# Arguments
- `nu::Integer`: Bessel order (only `0` is supported).
- `x::TaylorSeries.Taylor1{T}`: Taylor input.

# Returns
- `TaylorSeries.Taylor1{T}`: Taylor expansion of ``I_0(x)`` up to order `x.order`.

# Notes
- This method is implemented locally for the F0000 integrand module and is
  not intended as a global extension of `SpecialFunctions`.
- The construction assumes small Taylor perturbations around ``x_0``, which is
  the typical situation when Taylor objects arise in derivative-based error
  estimation.
"""
function besseli(nu::Integer, x::TaylorSeries.Taylor1{T}) where {T<:Real}
    nu == 0 || throw(ArgumentError("besseli(Taylor1) is implemented only for nu=0 (got nu=$nu)"))

    N  = x.order
    x0 = x.coeffs[1]
    dx = x - x0

    # Coefficients for Σ c[k+1] * dx^k
    c = Vector{T}(undef, N + 1)

    # k = 0
    c[1] = besseli(0, x0)

    # k >= 1
    g = z -> besseli(0, z)
    for k in 1:N
        dk = _nth_derivative_forwarddiff_scalar(g, x0, k)
        c[k + 1] = dk / factorial(k)
    end

    return TaylorSeries.Taylor1(c, N)
end

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
    x0 = float(_const_term(x))   # branch decision in Real only

    if x0 ≤ 50.0
        return exp(-x) * besseli(0, x)
    else
        invx = inv(x)

        s = one(T) +
            invx / T(8) +
            T(9) * invx^2 / T(128) +
            T(225) * invx^3 / T(3072) +
            T(11025) * invx^4 / T(98304) # +
            # T(893025) * invx^5 / T(3932160) +
            # T(108056025) * invx^6 / T(188743680)

        return s / sqrt(T(2) * T(pi) * x)
    end
    # return Bessels.besseli0x(x)
end

"""
    exI0_safe(
        x::T
    ) where {T<:Number}

Return the scaled modified Bessel factor ``exp(-x) \\, I_0(x)`` in a numerically safe way.

Uses a simple branch on the **real constant term** of ``x``:
- for ``x \\le 50``: evaluates ``\\exp(-x) \\; \\texttt{besseli}(0, x)`` directly,
- for ``x > 50``: uses a truncated large-`x` asymptotic expansion of ``exp(-x) \\, I_0(x)``.

This is written to work with plain reals as well as AD / Taylor types by avoiding
type-dependent branching.
"""
@inline function exI0_safe(
    x::FastDifferentiation.Node
)
    T = typeof(x)

    # Asymptotic expansion of exp(-x) I0(x):
    # exp(-x) I0(x) ~ 1/sqrt(2πx) * (1 + 1/(8x) + 9/(128x^2) + 225/(3072x^3) + 11025/(98304x^4) + ...)
    invx = inv(x)

    s = one(T) +
        invx / T(8) +
        T(9) * invx^2 / T(128) +
        T(225) * invx^3 / T(3072) +
        T(11025) * invx^4 / T(98304) # +
        # T(893025) * invx^5 / T(3932160) +
        # T(108056025) * invx^6 / T(188743680)

    return s / sqrt(T(2) * T(pi) * x)
    # return Bessels.besseli0x(x)
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
    if y == zero(T)
        return T(1) / T(2)
    elseif y == one(T)
        return zero(T)
    end

    x = (one(T) - y) / y  # x>0

    exI0 = exI0_safe(x)

    termA = (4T(pi)^2) * ((one(T) - y) / (y^3)) * (exI0^4)

    emx2 = exp(-x / T(2))
    bracket = one(T) - (T(1)/T(2)) * (one(T) + one(T)/y) * emx2
    termB = - (one(T) / (y * (one(T) - y))) * bracket

    return termA + termB
end

"""
    g_F0000_raw(
        y::T
    ) where {T<:Number}

Evaluate the **raw** F0000 integrand on ``y \\in (0, 1)``.

Computes ``x = \\dfrac{1 - y}{y}`` and returns the sum of:
- a term proportional to ``\\left(\\exp(-x) \\, I_0(x)\\right)^4`` with rational prefactors in ``y``,
- a correction term involving ``\\exp(-x/2)``.

This raw form has endpoint-singular prefactors, so it is intended for interior
evaluation (endpoint handling is done elsewhere).
"""
function g_F0000_raw(
    y::FastDifferentiation.Node
)
    T = typeof(y)

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

# Implementation note
When `t` is a `TaylorSeries.Taylor1` object (used for local derivative
extraction), the hard endpoint cutoff is intentionally skipped. Applying the
cutoff to a Taylor object would collapse the series to zero and destroy the
derivative information. In this case the function returns the analytic
expression `p * t^(p-1) * g_F0000_raw(t^p)` directly.

# Arguments
- `t::T`: Parameter in `[0, 1]`.

# Keyword arguments
- `p::Int=2`: Power used in the substitution `y = t^p`.
- `eps::T=T(1e-15)`: Endpoint cutoff. If `t ≤ eps` or `1 - t ≤ eps`, returns `0`.

# Returns
- `T`: The transformed integrand value `g̃(t)` (or zero near endpoints).

# Notes
- Endpoint suppression uses a hard cutoff only for ordinary numeric inputs.
  For `TaylorSeries.Taylor1` inputs (derivative extraction), the cutoff is
  intentionally skipped and the analytic transformed expression is returned.
- The cutoff decision is made using the real constant term of `t` and `eps`,
  to avoid type-dependent branching for AD / Taylor inputs. 

 # Errors
- Throws an error if `p < 1` (implicitly, via `t^(p-1)` or user intent); no
  explicit check is performed here to preserve original behavior.
"""
function gtilde_F0000(
    t::T; 
    p::Int=2, 
    eps::T=T(1e-15)
) where {T<:Number}

    # If t is a Taylor object (used only for local derivative extraction),
    # do NOT apply hard endpoint cutoffs, otherwise all derivatives collapse to 0.
    if t isa TaylorSeries.Taylor1
        y = t^p
        return T(p) * t^(p-1) * g_F0000_raw(y)
    end

    t0   = float(_const_term(t))     # Real
    eps0 = float(_const_term(eps))   # Real

    if t0 ≤ eps0
        return zero(T)
    elseif (1.0 - t0) ≤ eps0
        return zero(T)
    end    

    y = t^p
    return T(p) * t^(p-1) * g_F0000_raw(y)
    # return g_F0000_raw(t)
end

function g_F0000_pure(
    y::T
) where {T<:AbstractFloat}
    y == zero(T) && return T(1)/T(2)
    y == one(T)  && return zero(T)

    x = (one(T) - y) / y
    eI0 = besseli0x(x)

    termA = T(4) * T(pi)^2 * x * eI0^4
    termB = - (one(T) - (one(T) + x/T(2)) * exp(-x/T(2))) / x

    return (termA + termB) / y^2
end

function g_F0000_5d(
    y::T,
    θ1::T,
    θ2::T,
    θ3::T,
    θ4::T,
) where {T<:AbstractFloat}

    πT = T(pi)

    # ------------------------------------------------------------
    # Endpoint prescription
    # ------------------------------------------------------------
    # The original 1D integrand has removable / special endpoint values:
    #   g_F0000_pure(0) = 1/2
    #   g_F0000_pure(1) = 0
    #
    # In the 5D representation, we distribute that scalar endpoint value
    # uniformly over the four dummy angular directions, so that integrating
    # over θ1..θ4 still reproduces the same endpoint contribution.
    # ------------------------------------------------------------
    if y == zero(T)
        return (T(1) / T(2)) / πT^4
    elseif y == one(T)
        return zero(T)
    end

    x = (one(T) - y) / y

    s = (one(T) - cos(θ1)) +
        (one(T) - cos(θ2)) +
        (one(T) - cos(θ3)) +
        (one(T) - cos(θ4))

    # True 5D representation of
    #   4π^2 x * besseli0x(x)^4
    termA_5d = (T(4) * x / πT^2) * exp(-x * s)

    # Fake 5D lifting of the B-term:
    # constant in θ1..θ4, normalized by 1/π^4 so that
    # integration over θ1..θ4 reproduces the original 1D term.
    termB_1d = -(
        one(T) - (one(T) + x / T(2)) * exp(-x / T(2))
    ) / x

    termB_5d = termB_1d / πT^4

    return (termA_5d + termB_5d) / y^2
end