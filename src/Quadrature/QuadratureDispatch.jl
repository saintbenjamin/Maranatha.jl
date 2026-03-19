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
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

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

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a, b, N, rule, boundary; λ = λT, real_type = T
    )

    total = zero(T)
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
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

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

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a, b, N, rule, boundary; λ = λT, real_type = T
    )
    ys, wy = xs, wx

    total = zero(T)

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
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

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

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a, b, N, rule, boundary; λ = λT, real_type = T
    )
    ys, wy = xs, wx
    zs, wz = xs, wx

    total = zero(T)

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
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

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

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a, b, N, rule, boundary; λ = λT, real_type = T
    )
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    total = zero(T)

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
        dim::Int,
        λ = nothing,
        real_type = nothing,
    ) -> Real

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

# Keyword arguments
- `dim::Int`:
  Number of dimensions.
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a, b, N, rule, boundary; λ = λT, real_type = T
    )

    idx = ones(Int, dim)

    total = zero(T)
    args = Vector{T}(undef, dim)

    @inbounds while true
        wprod = one(T)
        for d in 1:dim
            i = idx[d]
            args[d] = xs[i]
            wprod *= ws[i]
        end

        iszero(wprod) || (total += wprod * f(args...))

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
        boundary;
        λ = nothing,
        use_cuda::Bool = false,
        threaded_subgrid::Bool = false,
        real_type = nothing,
    ) -> Real

Unified public dispatcher for tensor-product quadrature evaluation.

# Function description
This function provides the main quadrature entry point for `Maranatha`.
It selects one of three execution paths:

- CUDA-based quadrature when `use_cuda == true`,
- CPU threaded-subgrid quadrature when `threaded_subgrid == true`,
- otherwise the dimension-specialized local quadrature path
  ([`quadrature_1d`](@ref), [`quadrature_2d`](@ref),
  [`quadrature_3d`](@ref), [`quadrature_4d`](@ref), or
  [`quadrature_nd`](@ref)).

The optional parameter `λ` is forwarded mainly for smoothing B-spline rules,
and `real_type` controls the internal scalar type used for conversion and
accumulation.

# Arguments
- `integrand`:
  Callable integrand accepting `dim` scalar positional arguments.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
- `N`:
  Number of subdivisions or composite blocks per axis.
- `dim`:
  Number of dimensions.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `use_cuda::Bool = false`:
  If `true`, dispatch to the CUDA quadrature backend.
- `threaded_subgrid::Bool = false`:
  If `true`, dispatch to the CPU threaded-subgrid backend.
  This option is ignored when `use_cuda == true`.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and backend
  evaluation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

# Errors
- Propagates validation and backend errors from the selected execution path.
- Throws `ArgumentError` indirectly if the selected backend rejects the supplied
  dimensionality or rule configuration.

# Notes
- Backend priority is `use_cuda` first, then `threaded_subgrid`, then the local
  dimension-dispatched CPU implementation.
- This dispatcher performs strategy selection only; it does not implement a
  separate quadrature rule of its own.
"""
function quadrature(
    integrand,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ = nothing,
    use_cuda::Bool = false,
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    if use_cuda
        return QuadratureDispatchCUDA.quadrature_cuda(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            λ = λT,
            real_type = T,
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
            λ = λT,
            real_type = T,
        )
    end

    if dim == 1
        return quadrature_1d(integrand, a, b, N, rule, boundary; λ = λT, real_type = T)
    elseif dim == 2
        return quadrature_2d(integrand, a, b, N, rule, boundary; λ = λT, real_type = T)
    elseif dim == 3
        return quadrature_3d(integrand, a, b, N, rule, boundary; λ = λT, real_type = T)
    elseif dim == 4
        return quadrature_4d(integrand, a, b, N, rule, boundary; λ = λT, real_type = T)
    else
        return quadrature_nd(
            integrand, a, b, N, rule, boundary;
            λ = λT,
            dim = dim,
            real_type = T,
        )
    end
end

end  # module QuadratureDispatch