# ============================================================================
# src/rules/Integrate/integrate_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 3D tensor-product quadrature
# ============================================================

"""
    integrate_3d(
        f, 
        a, 
        b, 
        N, 
        rule
    ) -> Float64

Evaluate a ``3``-dimensional integral of ``f(x, y, z)`` over the cube domain `[a, b]^3`
using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using
[`quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule)` and forms the tensor product:
```math
\\sum_i \\sum_j \\sum_k w_i w_j w_k \\, f(x_i, y_j, z_k) \\,.
```
Loop ordering and accumulation are preserved exactly as implemented.

# Arguments
- `f`: Integrand callable `f(x, y, z)`.
- `a`, `b`: Cube domain bounds (used for all axes).
- `N`: Number of intervals per axis.
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.
"""
function integrate_3d(
    f, 
    a, 
    b, 
    N, 
    rule
)

    xs, wx = quadrature_1d_nodes_weights(a, b, N, rule)
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
                total += wij * wz[k] * f(xi, yj, zs[k])
            end
        end
    end

    return total
end