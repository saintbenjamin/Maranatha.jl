# ============================================================================
# src/Quadrature/QuadratureDispatch.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module QuadratureDispatch

import ..JobLoggerTools
import ..QuadratureUtils
import ..QuadratureNodes

include("QuadratureDispatch/QuadratureDispatchThreadedSubgrid.jl")
include("QuadratureDispatch/QuadratureDispatchCUDA.jl")

using .QuadratureDispatchThreadedSubgrid
using .QuadratureDispatchCUDA

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
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and computes:
```math
\\sum_i w_i f(x_i).
```

Rule-specific validation is centralized in the node/weight generator.

# Arguments
- `f`: Integrand callable ``f(x)``.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals / blocks.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_1d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    λ::Float64 = 0.0
)
    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)

    total = 0.0
    @inbounds for j in eachindex(xs)
        iszero(ws[j]) && continue
        val = f(xs[j])
        total += ws[j] * val
    end
    return total
end

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
# Function description
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms
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
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_2d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    λ::Float64 = 0.0
)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
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
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms
the tensor-product sum:
```math
\\sum_i \\sum_j \\sum_k w_i w_j w_k f(x_i, y_j, z_k).
```

# Arguments
- `f`: Integrand callable ``f(x, y, z)``.
- `a`, `b`: Bounds used on all axes.
- `N`: Number of intervals / blocks per axis.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_3d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    λ::Float64 = 0.0
)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
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

"""
    quadrature_4d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary
    ) -> Float64

Evaluate a ``4``-dimensional tensor-product quadrature over ``[a,b]^4``.

# Function description
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and forms
the tensor-product sum:
```math
\\sum_i \\sum_j \\sum_k \\sum_\\ell w_i w_j w_k w_\\ell f(x_i, y_j, z_k, t_\\ell) \\, .
```

# Arguments
- `f`: Integrand callable ``f(x, y, z, t)``.
- `a`, `b`: Bounds used on all axes.
- `N`: Number of intervals / blocks per axis.
- `rule`: Integration rule symbol.
- `boundary`: Boundary pattern symbol.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_4d(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    λ::Float64 = 0.0
)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    total = 0.0
    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            yj = ys[j]
            wij = wi * wy[j]
            for k in eachindex(zs)
                zk = zs[k]
                wijk = wij * wz[k]
                for l in eachindex(ts)
                    w = wijk * wt[l]
                    iszero(w) && continue
                    val = f(xi, yj, zk, ts[l])
                    total += w * val
                end
            end
        end
    end
    return total
end

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
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)`, then
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
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_nd(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    dim::Int,
    λ::Float64 = 0.0
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)

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

"""
    quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary
    ) -> Float64

Evaluate a tensor-product quadrature on the hypercube ``[a,b]^{\\texttt{dim}}``.

# Function description
This is the unified integration dispatcher for the quadrature layer.

It obtains the underlying ``1``-dimensional nodes and weights via
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref), then chooses a dimension-specific
tensor-product evaluator:

- [`quadrature_1d`](@ref) for `dim == 1`
- [`quadrature_2d`](@ref) for `dim == 2`
- [`quadrature_3d`](@ref) for `dim == 3`
- [`quadrature_4d`](@ref) for `dim == 4`
- [`quadrature_nd`](@ref) otherwise

All axes use the same interval ``[a,b]``, so the integration domain is the
hypercube ``[a,b]^{\\texttt{dim}}``.

# Arguments
- `integrand`: Callable accepting exactly `dim` positional arguments.
- `a`, `b`: Lower and upper bounds applied to every axis.
- `N`: Number of subintervals / blocks per axis.
- `dim`: Number of dimensions.
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern selector.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Throws an error if ``\\texttt{dim} < 1``.
- Throws any rule-validation or backend error propagated from the selected
  quadrature generator.
- Propagates any error thrown by `integrand`.
"""
function quadrature(
    integrand,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ::Float64 = 0.0,
    use_cuda::Bool = false,
    threaded_subgrid::Bool = false,
)
    if use_cuda
        return QuadratureDispatchCUDA.quadrature_cuda(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            λ=λ
        )
    end

    if threaded_subgrid
        return QuadratureDispatchThreadedSubgrid.quadrature_threaded_subgrid(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            λ=λ
        )
    end

    if dim == 1
        return quadrature_1d(integrand, a, b, N, rule, boundary; λ=λ)
    elseif dim == 2
        return quadrature_2d(integrand, a, b, N, rule, boundary; λ=λ)
    elseif dim == 3
        return quadrature_3d(integrand, a, b, N, rule, boundary; λ=λ)
    elseif dim == 4
        return quadrature_4d(integrand, a, b, N, rule, boundary; λ=λ)
    else
        return quadrature_nd(integrand, a, b, N, rule, boundary; λ=λ, dim = dim)
    end
end

end  # module QuadratureDispatch