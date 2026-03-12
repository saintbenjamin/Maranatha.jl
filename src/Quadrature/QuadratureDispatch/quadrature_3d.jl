# ============================================================================
# src/Quadrature/QuadratureDispatch/quadrature_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_3d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate a ``3``-dimensional tensor-product quadrature over ``[a,b]^3``.

# Function description
This routine generates `1`-dimensional nodes and weights using
[`get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms
the tensor-product sum:
```math
\\sum_i \\sum_j \\sum_k w_i w_j w_k f(x_i, y_j, z_k).
```

# Arguments
- `f`: Integrand callable `f(x, y, z)`.
- `a`, `b`: Bounds used on all axes.
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
function quadrature_3d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary
)

    xs, wx = get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)
    ys, wy = xs, wx
    zs, wz = xs, wx

    total = 0.0

    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            yj = ys[j]
            wij = wi * wy[j]
            for k in eachindex(zs)
                w = wij * wz[k]
                iszero(w) && continue
                val = f(xi, yj, zs[k])
                total += w * val
            end
        end
    end

    return total
end