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

Evaluate a ``1``-dimensional quadrature over ``[a,b]``.

# Function description
This routine generates `1`-dimensional nodes and weights using
[`get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and computes:
```math
\\sum_i w_i f(x_i).
```

Rule-specific validation is centralized in the node/weight generator.

# Arguments
- `f`: Integrand callable `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals / blocks.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Propagates any error thrown by
  [`get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
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