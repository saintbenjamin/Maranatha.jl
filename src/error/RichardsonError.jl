# ============================================================================
# src/error/RichardsonError.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module RichardsonError

using ..Integrate

export estimate_error_richardson

"""
    rule_order(
        rule::Symbol
    ) -> Int

Return the leading convergence order `p` of the specified Newton–Cotes rule,
assuming an asymptotic error scaling of `O(h^p)`.

# Function description
This function maps each supported integration rule to its expected leading
error order used in Richardson extrapolation. The mapping is intentionally
kept identical to the original implementation so that numerical behavior
remains unchanged.

# Arguments
- `rule`: Integration rule symbol.

# Supported rules and orders
- `:simpson13_close` → `p = 4`
- `:simpson38_close` → `p = 4`
- `:bode_close`      → `p = 6`
- `:simpson13_open`  → `p = 4`
- `:simpson38_open`  → `p = 4`
- `:bode_open`       → `p = 6`

# Returns
- Integer convergence order `p`.

# Errors
- Throws an error if `rule` is not supported.
"""
function rule_order(
    rule::Symbol
)
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
    estimate_error_richardson(
        integrand, 
        a, 
        b, 
        N::Int, 
        dim::Int, 
        rule::Symbol
    ) -> Float64

Estimate the integration error using a Richardson-based scale comparison
between resolutions `N` and `2N`.

# Function description
This estimator evaluates the integral twice using the same rule:
- `I(N)`  with resolution `N`
- `I(2N)` with resolution `2N`

Assuming a leading-order error scaling of `O(h^p)`, the Richardson estimate is

```julia
err ≈ |I(2N) - I(N)| / (2^p - 1)
```
where `p = rule_order(rule)`.

This implementation intentionally preserves the exact call order and
floating-point behavior of the original version.

# Arguments
- `integrand`: Function representing the integrand. Must accept `dim` positional arguments.
- `a`, `b`: Scalar bounds defining the hypercube `[a,b]^dim`.
- `N`: Base resolution (number of subintervals per axis).
- `dim`: Number of dimensions (forwarded to `integrate_nd`).
- `rule`: Integration rule symbol.

# Returns
- A `Float64` Richardson-style error scale computed from `I(N)` and `I(2N)`.

# Notes
- This provides an *error scale* rather than a strict bound.
- Any rule-specific constraints on `N` (e.g., divisibility) must be satisfied
  by both `N` and `2N`.
"""
function estimate_error_richardson(
    integrand, 
    a, 
    b, 
    N::Int, 
    dim::Int, 
    rule::Symbol
)
    p   = rule_order(rule)
    I_N  = integrate_nd(integrand, a, b, N,  dim, rule)
    I_2N = integrate_nd(integrand, a, b, 2N, dim, rule)
    return abs(I_2N - I_N) / (2.0^p - 1.0)
end

end  # module RichardsonError