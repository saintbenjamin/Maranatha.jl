module RichardsonError

using ..Integrate

export estimate_error_richardson

# leading order: error ~ O(h^p)
function rule_order(rule::Symbol)
    if rule == :simpson13_close || rule == :simpson38_close
        return 4
    elseif rule == :bode_close
        return 6
    elseif rule == :simpson13_open
        return 4
    elseif rule == :simpson38_open
        return 4
    elseif rule == :bode_open
        return 6
    else
        error("rule_order: unsupported rule = $rule")
    end
end

"""
    estimate_error_richardson(integrand, a, b, N, dim, rule) -> Float64

Richardson-based error scale using I(N) and I(2N):
    err ≈ |I(2N) - I(N)| / (2^p - 1)
where p = rule_order(rule).
"""
function estimate_error_richardson(integrand, a, b, N::Int, dim::Int, rule::Symbol)
    p   = rule_order(rule)
    I_N  = integrate_nd(integrand, a, b, N,  dim, rule)
    I_2N = integrate_nd(integrand, a, b, 2N, dim, rule)
    return abs(I_2N - I_N) / (2.0^p - 1.0)
end

end