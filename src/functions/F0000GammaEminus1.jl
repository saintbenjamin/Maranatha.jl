module F0000GammaEminus1

# =========================
# F0000 integral via Maranatha (1D quadrature on y ∈ [0, 1])
# =========================
using SpecialFunctions

export gtilde_F0000

# exp(-x) * I0(x), overflow-safe
# - for small/moderate x: direct evaluation
# - for large x: asymptotic expansion (no overflow)
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

function g_F0000_raw(y::T) where {T<:Real}
    x = (one(T) - y) / y  # x>0

    exI0 = exI0_safe(x)

    termA = (4T(pi)^2) * ((one(T) - y) / (y^3)) * (exI0^4)

    emx2 = exp(-x / T(2))
    bracket = one(T) - (T(1)/T(2)) * (one(T) + one(T)/y) * emx2
    termB = - (one(T) / (y * (one(T) - y))) * bracket

    return termA + termB
end

function gtilde_F0000(t::T; p::Int=2, eps::T=T(1e-15)) where {T<:Real}
    if t ≤ eps
        return zero(T)
    elseif (one(T) - t) ≤ eps
        return zero(T)
    end

    y = t^p
    return T(p) * t^(p-1) * g_F0000_raw(y)
end

end