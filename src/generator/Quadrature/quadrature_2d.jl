# ============================================================================
# src/rules/Integrate/integrate_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    integrate_2d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate a ``2``-dimensional integral of ``f(x, y)`` over the square domain ``[a, b] \\times [a, b]``
using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using
[`quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms the tensor product:
```math
\\sum_i \\sum_j w_i w_j \\, f(x_i, y_j) \\,.
```
Loop ordering and accumulation are preserved exactly as implemented.

# Arguments
- `f`: Integrand callable `f(x, y)`.
- `a`, `b`: Square domain bounds (used for both axes).
- `N`: Number of intervals per axis.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
  Required for NS rules.

# Returns
- Estimated integral value as a `Float64`.
"""
function integrate_2d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary
)

    xs, wx = quadrature_1d_nodes_weights(a, b, N, rule, boundary)
    ys, wy = xs, wx   # same bounds

    total = 0.0

    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            w = wi * wy[j]
            iszero(w) && continue
            val = f(xi, ys[j])
            total += w * val
        end
    end

    return total
end