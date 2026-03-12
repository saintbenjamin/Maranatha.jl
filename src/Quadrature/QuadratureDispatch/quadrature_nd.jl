# ============================================================================
# src/Quadrature/QuadratureDispatch/quadrature_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_nd(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int
    ) -> Float64

Perform a general tensor-product quadrature over ``[a,b]^{\\texttt{dim}}``.

# Function description
This routine builds the `1`-dimensional nodes and weights using
[`get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)`, then
enumerates all tensor-product index tuples with an odometer-style update.

For each multi-index ``(i_1, \\ldots, i_{\\texttt{dim}})``, it forms the weight
product and evaluates the integrand as ``f(x_1, x_2, \\ldots, x_{\\texttt{dim}})`` using splatting.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Bounds defining the hypercube ``[a,b]^{\\texttt{dim}}``.
- `N`: Number of subdivisions / blocks per axis.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.
- `dim`: Number of dimensions.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Throws `ArgumentError` if ``\\texttt{dim} < 1``.
- Propagates any error thrown by
  [`get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_nd(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    xs, ws = get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)

    # Multi-index over axes (1-based)
    idx = ones(Int, dim)

    total = 0.0
    args = Vector{Float64}(undef, dim)

    @inbounds while true
        wprod = 1.0
        for d in 1:dim
            i = idx[d]
            args[d] = xs[i]
            wprod *= ws[i]
        end

        # Call f(x1, x2, ..., x_dim)
        iszero(wprod) || (total += wprod * f(args...))

        # Increment odometer-style index
        d = dim
        while d >= 1
            idx[d] += 1
            if idx[d] <= length(xs)
                break
            else
                idx[d] = 1
                d -= 1
            end
        end
        d == 0 && break
    end

    return total
end