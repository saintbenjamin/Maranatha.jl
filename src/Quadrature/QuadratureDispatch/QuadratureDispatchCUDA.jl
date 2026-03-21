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
import ..QuadratureBoundarySpec
import ..QuadratureRuleSpec
import ..QuadratureNodes

"""
    _linear_to_indices_cuda(
        q::Int,
        lens::NTuple{D,Int},
        ::Val{D},
    ) where {D}

Convert a zero-based linear tensor-product index into a `D`-dimensional tuple
of one-based quadrature-node indices.

# Function description
This generated helper decodes the flattened tensor-product index `q` into the
corresponding per-axis indices for a `D`-dimensional quadrature grid, where the
number of quadrature nodes may differ from axis to axis according to `lens`.
The returned indices are one-based so they can be used directly for Julia array access.

Conceptually, this performs a mixed-radix decomposition of `q` using the axis
lengths in `lens`, then shifts each digit by `+1` to match Julia indexing.

# Arguments
- `q::Int`:
  Zero-based flattened tensor-product index.
- `lens::NTuple{D,Int}`:
  Number of quadrature nodes on each axis.
- `::Val{D}`:
  Compile-time dimensionality tag.

# Returns
- `NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.

# Notes
- This helper is generated so the tuple size and index extraction logic are
  specialized for each dimension `D`.
- The input `q` is assumed to satisfy `0 <= q < prod(lens)`.
"""
@generated function _linear_to_indices_cuda(
    q::Int,
    lens::NTuple{D,Int},
    ::Val{D},
) where {D}
    vars = [Symbol(:i, d) for d in 1:D]
    body = Expr(:block, :(qq = q))
    for d in D:-1:1
        push!(body.args, :($(vars[d]) = (qq % lens[$d]) + 1))
        if d > 1
            push!(body.args, :(qq ÷= lens[$d]))
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
- `ws_mat`:
  Matrix of one-dimensional quadrature weights, with one column per axis.
- `idxs::NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.

# Returns
- The tensor-product weight formed as
  `ws_mat[idxs[1], 1] * ws_mat[idxs[2], 2] * ... * ws_mat[idxs[D], D]`.

# Notes
- This helper supports axis-wise quadrature rules represented as one column per axis.
- It is generated to unroll the product at compile time for each dimension `D`.
"""
@generated function _weight_product_cuda(
    ws_mat,
    idxs::NTuple{D,Int},
    ::Val{D},
) where {D}
    ex = :(ws_mat[idxs[1], 1])
    for d in 2:D
        ex = :($ex * ws_mat[idxs[$d], $d])
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
This generated helper gathers the coordinate values from `xs_mat` using the
multi-dimensional index tuple `idxs`, then calls the integrand `f` with those
coordinates as `D` positional arguments.
Each axis reads from its own column, so per-axis node grids are supported.

For example, in 3D it produces a call equivalent to

```julia
f(xs_mat[idxs[1], 1], xs_mat[idxs[2], 2], xs_mat[idxs[3], 3])
```

# Arguments

* `f`:
  Integrand callable.
* `xs_mat`:
  Matrix of one-dimensional quadrature nodes, with one column per axis.
* `idxs::NTuple{D,Int}`:
  One-based quadrature-node indices for each axis.
* `::Val{D}`:
  Compile-time dimensionality tag.

# Returns

* The value of `f` evaluated at the selected tensor-product node.

# Notes

* This helper supports axis-wise node sets by reading coordinates from the
  corresponding axis column of `xs_mat`.
* It is generated so the positional call to `f` is fully specialized for `D`.
"""
@generated function _eval_f_cuda(
    f,
    xs_mat,
    idxs::NTuple{D,Int},
    ::Val{D},
) where {D}
    args = [:(xs_mat[idxs[$d], $d]) for d in 1:D]
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
* `xs_mat`:
  Device matrix of one-dimensional quadrature nodes, with one column per axis.
* `ws_mat`:
  Device matrix of one-dimensional quadrature weights, with one column per axis.
* `lens::NTuple{D,Int}`:
  Number of valid quadrature nodes on each axis.
* `total_points`:
  Total number of tensor-product quadrature points, typically `prod(lens)`.
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
* Rectangular axis-wise domains are supported by allowing each axis to have its
  own node and weight column and its own valid node count in `lens`.
"""
function _kernel_quadrature_nd!(
    out,
    xs_mat,
    ws_mat,
    lens::NTuple{D,Int},
    total_points::Int,
    ::Val{D},
    f,
) where {D}
    idx =
        (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x

    stride =
        CUDA.gridDim().x * CUDA.blockDim().x

    total = zero(eltype(out))

    for linear in idx:stride:total_points
        q = linear - 1
        idxs = _linear_to_indices_cuda(q, lens, Val(D))

        w = _weight_product_cuda(ws_mat, idxs, Val(D))

        if !iszero(w)
            total += w * _eval_f_cuda(f, xs_mat, idxs, Val(D))
        end
    end

    out[idx] = total
    return
end

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
        λ = nothing,
        real_type = nothing,
    )

Evaluate a CUDA-based tensor-product quadrature approximation of `f`.

# Function description

This is the public GPU quadrature entry point of `QuadratureDispatchCUDA`.
It supports both of the following domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval `[a, b]` is used on every axis.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, each axis uses its own
  interval `[a[d], b[d]]`.

The routine constructs one-dimensional quadrature nodes and weights for each
axis using the requested rule and boundary selector, transfers them to CUDA
device memory, launches a CUDA kernel over the full tensor-product grid, and
then sums the per-thread partial contributions on the host.

# Arguments

* `f`:
  CUDA-callable integrand accepting `dim` positional arguments.
* `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds of length `dim`.
* `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds of length `dim`.
* `N`:
  Quadrature subdivision or rule-resolution parameter forwarded to
  `QuadratureNodes.get_quadrature_1d_nodes_weights`.
* `rule`:
  Quadrature rule specification. This may be either a scalar rule symbol
  shared across all axes or a tuple/vector of per-axis rule symbols of length
  `dim`.
* `boundary`:
  Boundary specification. This may be either a scalar boundary symbol shared
  across all axes or a tuple/vector of per-axis boundary symbols of length
  `dim`.
* `dim::Int`:
  Number of dimensions.
* `threads::Int = 256`:
  Number of CUDA threads per block.
* `λ = nothing`:
  Optional extra rule parameter forwarded to the node/weight generator.
  If `nothing`, zero is used in the active scalar type.
* `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and CUDA
  execution. CUDA mode currently supports only `Float32` and `Float64`.
  If omitted, the type is inferred from scalar bounds, or from all components
  of axis-wise bounds.

# Returns

* `Real`:
  Final quadrature approximation obtained from the CUDA backend, in the active scalar type.

# Errors

* Throws `ArgumentError` if `dim < 1`.
* Throws `ArgumentError` if `threads < 1`.
* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
* Throws `ArgumentError` if the active scalar type is not `Float32` or `Float64`.
* Propagates errors from `QuadratureNodes.get_quadrature_1d_nodes_weights`.
* Propagates CUDA kernel launch or execution errors.

# Notes

* This implementation supports both isotropic hypercube domains and rectangular
  axis-wise domains.
* The total number of tensor-product points is `prod(lens)`, where `lens[d]`
  is the number of quadrature nodes on axis `d`.
* Internally, per-axis node and weight vectors are packed into dense matrices
  before transfer to the GPU.
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
    λ = nothing,
    real_type = nothing,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    threads >= 1 || throw(ArgumentError("threads must be ≥ 1"))

    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end

    (T === Float32 || T === Float64) || throw(ArgumentError(
        "CUDA mode currently supports only Float32 or Float64 real_type (got $(T))."
    ))

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    xs_list = Vector{Vector{T}}(undef, dim)
    ws_list = Vector{Vector{T}}(undef, dim)

    if !(a isa AbstractVector || a isa Tuple)
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a,
                b,
                N,
                rule,
                boundary;
                λ = λT,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    else
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a[d],
                b[d],
                N,
                rule,
                boundary;
                λ = λT,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    end

    lens_vec = [length(xs_list[d]) for d in 1:dim]
    maxn = maximum(lens_vec)

    xs_mat = zeros(T, maxn, dim)
    ws_mat = zeros(T, maxn, dim)

    for d in 1:dim
        nd = lens_vec[d]
        xs_mat[1:nd, d] .= xs_list[d]
        ws_mat[1:nd, d] .= ws_list[d]
    end

    xs_d = CUDA.CuArray(xs_mat)
    ws_d = CUDA.CuArray(ws_mat)

    lens = ntuple(d -> lens_vec[d], dim)
    total_points = prod(lens_vec)
    blocks = cld(total_points, threads)

    out = CUDA.zeros(T, threads * blocks)

    JobLoggerTools.println_benji(
        "CUDA backend: dim=$(dim), lens=$(lens_vec) → total points=$(total_points) | blocks=$(blocks), threads/block=$(threads)"
    )

    CUDA.@cuda threads=threads blocks=blocks _kernel_quadrature_nd!(
        out,
        xs_d,
        ws_d,
        lens,
        total_points,
        Val(dim),
        f,
    )

    return sum(Array(out))
end

end  # module QuadratureDispatchCUDA
