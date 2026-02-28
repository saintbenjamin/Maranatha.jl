# ============================================================================
# src/rules/Integrate/integrate_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    integrate_1d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate the ``1``-dimensional integral of ``f(x)`` over ``[a, b]`` using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using [`quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and computes:
```math
\\sum_i w_i \\, f(x_i) \\,.
```
This keeps all rule-specific constraints and behaviour centralized in
[`quadrature_1d_nodes_weights`](@ref).

# Arguments
- `f`: Integrand callable `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals (rule-specific constraints are enforced by [`quadrature_1d_nodes_weights`](@ref)).
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
  Required for NS rules.

# Returns
- Estimated integral value as a `Float64`.
"""
function integrate_1d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary
)
    xs, ws = quadrature_1d_nodes_weights(a, b, N, rule, boundary)

    total = 0.0
    @inbounds for j in eachindex(xs)
        iszero(ws[j]) && continue
        val = f(xs[j])
        total += ws[j] * val
    end
    return total
end