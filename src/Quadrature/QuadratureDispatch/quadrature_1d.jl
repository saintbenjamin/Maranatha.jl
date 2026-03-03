# ============================================================================
# src/Quadrature/QuadratureDispatch/quadrature_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_1d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate the ``1``-dimensional integral of ``f(x)`` over ``[a, b]`` using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using [`get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and computes:
```math
\\sum_i w_i \\, f(x_i) \\,.
```
This keeps all rule-specific constraints and behaviour centralized in
[`get_quadrature_1d_nodes_weights`](@ref).

# Arguments
- `f`: Integrand callable `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals (rule-specific constraints are enforced by [`get_quadrature_1d_nodes_weights`](@ref)).
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
  Required for NS rules.

# Returns
- Estimated integral value as a `Float64`.
"""
function quadrature_1d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary
)
    xs, ws = get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)

    total = 0.0
    @inbounds for j in eachindex(xs)
        iszero(ws[j]) && continue
        val = f(xs[j])
        total += ws[j] * val
    end
    return total
end