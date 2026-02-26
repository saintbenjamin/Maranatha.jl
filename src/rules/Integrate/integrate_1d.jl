# ============================================================================
# src/rules/Integrate/integrate_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 1D quadrature legacy
# ============================================================

"""
    integrate_1d_legacy(
        f, 
        a, 
        b, 
        N, 
        rule
    ) -> Float64

Evaluate a 1D integral of `f(x)` over `[a, b]` using the specified quadrature rule.

# Function description
This function dispatches to the dedicated 1D implementations for each supported
Newton–Cotes rule. Both closed and open-chain variants are supported.

# Arguments
- `f`: Integrand function `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals (rule-specific divisibility/minimum constraints apply).
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.

# Errors
- Throws an error if `rule` is not recognized.
"""
function integrate_1d_legacy(
    f, 
    a, 
    b, 
    N, 
    rule
)
    if rule == :simpson13_close
        return simpson13_rule(f, a, b, N)

    elseif rule == :simpson13_open
        return simpson13_rule_min_open_max_open(f, a, b, N)

    elseif rule == :simpson38_close
        return simpson38_rule(f, a, b, N)

    elseif rule == :simpson38_open
        return simpson38_rule_min_open_max_open(f, a, b, N)

    elseif rule == :bode_close
        return bode_rule(f, a, b, N)

    elseif rule == :bode_open
        return bode_rule_min_open_max_open(f, a, b, N)

    else
        JobLoggerTools.error_benji("Unknown integration rule: $rule")
    end
end

# ============================================================
# 1D quadrature current version
# ============================================================

"""
    integrate_1d(
        f, 
        a, 
        b, 
        N, 
        rule
    ) -> Float64

Evaluate a 1D integral of `f(x)` over `[a, b]` using the specified quadrature rule.

# Function description
This routine uses `quadrature_1d_nodes_weights(a, b, N, rule)` and computes:
`Σ_j w_j f(x_j)`.

This matches the tensor-product style used in `integrate_2d/3d/...` and keeps all
rule-specific constraints/behavior centralized in `quadrature_1d_nodes_weights`.

# Arguments
- `f`: Integrand callable `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals (rule-specific constraints are enforced by `quadrature_1d_nodes_weights`).
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.
"""
function integrate_1d(
    f, 
    a, 
    b, 
    N, 
    rule
)
    xs, ws = quadrature_1d_nodes_weights(a, b, N, rule)

    total = 0.0
    @inbounds for j in eachindex(xs)
        total += ws[j] * f(xs[j])
    end
    return total
end