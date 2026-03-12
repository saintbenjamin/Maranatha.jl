# ============================================================================
# src/Quadrature/QuadratureDispatch/quadrature_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_2d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate a ``2``-dimensional tensor-product quadrature over ``[a,b] \\times [a,b]``.

# Function description
This routine generates ``1``-dimensional nodes and weights using
[`get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms
the tensor-product sum:
```math
\\sum_i \\sum_j w_i w_j f(x_i, y_j).
```

# Arguments
- `f`: Integrand callable ``f(x, y)``.
- `a`, `b`: Bounds used on both axes.
- `N`: Number of intervals / blocks per axis.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Propagates any error thrown by
  [`get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_2d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary
)

    xs, wx = get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)
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