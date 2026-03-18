# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchCUDA.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureDispatchCUDA

CUDA-based tensor-product quadrature backend for `Maranatha.jl`.

# Module description
`QuadratureDispatchCUDA` provides a GPU-oriented quadrature path for
multi-dimensional tensor-product integration. It builds the one-dimensional
quadrature nodes and weights on the host, transfers them to CUDA device memory,
launches a CUDA kernel over the full tensor-product grid, and reduces the
partial thread-local contributions into a final scalar quadrature value.

# Responsibilities
- construct CUDA-compatible tensor-product quadrature evaluation
- map linear CUDA work indices to multi-dimensional quadrature-node indices
- evaluate the integrand on the GPU over the full tensor-product grid
- accumulate weighted quadrature contributions in parallel

# Notes
- This module assumes that the supplied integrand `f` is callable inside a CUDA
  kernel and is compatible with GPU execution constraints.
- The quadrature rule itself is still defined by the shared
  `QuadratureNodes.get_quadrature_1d_nodes_weights` interface.
- The public entry point of this module is [`quadrature_cuda`](@ref).
"""
module QuadratureDispatchCUDA

import CUDA

import ..JobLoggerTools
import ..QuadratureNodes

"""
    _linear_to_indices_cuda(
        q::Int,
        n::Int,
        ::Val{D},
    ) where {D}

Convert a zero-based linear tensor-product index into a `D`-dimensional tuple
of one-based quadrature-node indices.

# Function description
This generated helper decodes the flattened tensor-product index `q` into the
corresponding per-axis indices for a `D`-dimensional quadrature grid, where
each axis has `n` quadrature nodes. The returned indices are one-based so they
can be used directly for Julia array access.

Conceptually, this performs a base-`n` decomposition of `q`, then shifts each
digit by `+1` to match Julia indexing.

# Arguments
- `q::Int`:
  Zero-based flattened tensor-product index.
- `n::Int`:
  Number of quadrature nodes per axis.
- `::Val{D}`:
  Compile-time dimensionality tag.

# Returns
- `NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.

# Notes
- This helper is generated so the tuple size and index extraction logic are
  specialized for each dimension `D`.
- The input `q` is assumed to satisfy `0 <= q < n^D`.
"""
@generated function _linear_to_indices_cuda(
    q::Int,
    n::Int,
    ::Val{D},
) where {D}
    vars = [Symbol(:i, d) for d in 1:D]
    body = Expr(:block, :(qq = q))
    for d in D:-1:1
        push!(body.args, :($(vars[d]) = (qq % n) + 1))
        if d > 1
            push!(body.args, :(qq ÷= n))
        end
    end
    push!(body.args, :(return ($(vars...),)))
    return body
end

"""
    _weight_product_cuda(
        ws,
        idxs::NTuple{D,Int},
    ) where {D}

Compute the tensor-product quadrature weight associated with `idxs`.

# Function description
This generated helper multiplies the one-dimensional quadrature weights stored
in `ws` across all `D` axes using the node indices in `idxs`. The result is the
full tensor-product weight corresponding to a single multi-dimensional
quadrature point.

# Arguments
- `ws`:
  One-dimensional quadrature weights.
- `idxs::NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.

# Returns
- The tensor-product weight formed as
  `ws[idxs[1]] * ws[idxs[2]] * ... * ws[idxs[D]]`.

# Notes
- This helper assumes the same one-dimensional rule is used on every axis.
- It is generated to unroll the product at compile time for each dimension `D`.
"""
@generated function _weight_product_cuda(
    ws,
    idxs::NTuple{D,Int},
) where {D}
    ex = :(ws[idxs[1]])
    for d in 2:D
        ex = :($ex * ws[idxs[$d]])
    end
    return ex
end

"""
    _eval_f_cuda(
        f,
        xs,
        idxs::NTuple{D,Int},
        ::Val{D},
    ) where {D}

Evaluate the integrand `f` at the tensor-product quadrature node selected by
`idxs`.

# Function description
This generated helper gathers the coordinate values from `xs` using the
multi-dimensional index tuple `idxs`, then calls the integrand `f` with those
coordinates as `D` positional arguments.

For example, in 3D it produces a call equivalent to

```julia
f(xs[idxs[1]], xs[idxs[2]], xs[idxs[3]])
```

# Arguments

* `f`:
  Integrand callable.
* `xs`:
  One-dimensional quadrature nodes.
* `idxs::NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.
* `::Val{D}`:
  Compile-time dimensionality tag.

# Returns

* The value of `f` evaluated at the selected tensor-product node.

# Notes

* This helper assumes the same one-dimensional node vector `xs` is reused on
  every axis.
* It is generated so the positional call to `f` is fully specialized for `D`.
"""
@generated function _eval_f_cuda(
    f,
    xs,
    idxs::NTuple{D,Int},
    ::Val{D},
) where {D}
    args = [:(xs[idxs[$d]]) for d in 1:D]
    return :(f($(args...)))
end

"""
    _kernel_quadrature_nd!(
        out,
        xs,
        ws,
        n,
        total_points,
        ::Val{D},
        f,
    ) where {D}

CUDA kernel that evaluates a `D`-dimensional tensor-product quadrature sum.

# Function description

Each CUDA thread iterates over a strided subset of the flattened tensor-product
grid, converts each linear index into per-axis node indices, evaluates the
corresponding tensor-product weight and integrand value, and accumulates the
weighted contribution into a thread-local partial sum. The final partial sum for
that thread is written into `out[idx]`.

# Arguments

* `out`:
  Device output array storing one partial sum per launched CUDA thread.
* `xs`:
  Device array of one-dimensional quadrature nodes.
* `ws`:
  Device array of one-dimensional quadrature weights.
* `n`:
  Number of quadrature nodes per axis.
* `total_points`:
  Total number of tensor-product quadrature points, typically `n^D`.
* `::Val{D}`:
  Compile-time dimensionality tag.
* `f`:
  CUDA-callable integrand.

# Returns

* `nothing`

# Notes

* This kernel performs no global reduction; it writes thread-local partial sums
  into `out`, and the host later reduces them.
* The loop uses a standard CUDA grid-stride pattern.
* `out[idx]` assumes that the allocated output array length matches the total
  number of launched CUDA threads.
"""
function _kernel_quadrature_nd!(
    out,
    xs,
    ws,
    n,
    total_points,
    ::Val{D},
    f,
) where {D}
    idx =
        (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x

    stride =
        CUDA.gridDim().x * CUDA.blockDim().x

    total = 0.0

    for linear in idx:stride:total_points
        q = linear - 1
        idxs = _linear_to_indices_cuda(q, n, Val(D))

        w = _weight_product_cuda(ws, idxs)

        if !iszero(w)
            total += w * _eval_f_cuda(f, xs, idxs, Val(D))
        end
    end

    out[idx] = total
    return
end

# ============================================================
# Public API (single function, dim argument)
# ============================================================

"""
    quadrature_cuda(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        threads::Int = 256,
        λ::Float64 = 0.0,
    )

Evaluate a CUDA-based tensor-product quadrature approximation of `f` over a
`dim`-dimensional hypercube.

# Function description

This is the public GPU quadrature entry point of `QuadratureDispatchCUDA`.
It first constructs the one-dimensional quadrature nodes and weights on the
interval `[a, b]` using the requested rule and boundary selector, transfers
them to CUDA device memory, launches a CUDA kernel over the full tensor-product
grid, and then sums the per-thread partial contributions on the host.

# Arguments

* `f`:
  CUDA-callable integrand accepting `dim` positional arguments.
* `a`:
  Lower bound on each integration axis.
* `b`:
  Upper bound on each integration axis.
* `N`:
  Quadrature subdivision or rule-resolution parameter forwarded to
  `QuadratureNodes.get_quadrature_1d_nodes_weights`.
* `rule`:
  Quadrature rule symbol.
* `boundary`:
  Boundary-condition symbol.
* `dim::Int`:
  Number of dimensions.
* `threads::Int = 256`:
  Number of CUDA threads per block.
* `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns

* `Float64`:
  Final quadrature approximation obtained from the CUDA backend.

# Errors

* Throws `ArgumentError` if `dim < 1`.
* Throws `ArgumentError` if `threads < 1`.
* Propagates errors from `QuadratureNodes.get_quadrature_1d_nodes_weights`.
* Propagates CUDA kernel launch or execution errors.

# Notes

* This implementation assumes the same one-dimensional quadrature nodes and
  weights are used on every axis.
* The total number of tensor-product points is `n^dim`, where `n = length(xs)`.
* This routine is suitable only when `f` is GPU-compatible.
"""
function quadrature_cuda(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    threads::Int = 256,
    λ::Float64 = 0.0
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    threads >= 1 || throw(ArgumentError("threads must be ≥ 1"))

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    n = length(xs)

    xs_d = CUDA.CuArray(xs)
    ws_d = CUDA.CuArray(ws)

    total_points = n^dim
    blocks = cld(total_points, threads)

    out = CUDA.zeros(Float64, threads * blocks)

    JobLoggerTools.println_benji(
        "CUDA backend: dim=$(dim), n=$(n) → total points=$(total_points) | blocks=$(blocks), threads/block=$(threads)"
    )

    CUDA.@cuda threads=threads blocks=blocks _kernel_quadrature_nd!(
        out,
        xs_d,
        ws_d,
        n,
        total_points,
        Val(dim),
        f,
    )

    return sum(Array(out))
end

end  # module QuadratureDispatchCUDA