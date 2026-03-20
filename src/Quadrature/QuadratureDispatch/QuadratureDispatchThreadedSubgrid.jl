# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureDispatchThreadedSubgrid

Thread-based subgrid-partitioned tensor-product quadrature backend.

# Module description
`QuadratureDispatchThreadedSubgrid` provides a CPU multithreading backend for
tensor-product quadrature. Instead of parallelizing only the outermost loop,
this module partitions the full tensor-product grid into rectangular subblocks
across one or more axes, assigns those blocks to Julia threads, and reduces the
thread-local partial sums into a final quadrature value.

# Responsibilities
- split one-dimensional index ranges into balanced chunks
- choose per-axis split counts for a requested thread budget
- build rectangular tensor-product subgrid blocks from axis splits
- evaluate 1D, 2D, 3D, 4D, and generic ND quadrature using threaded subgrid
  traversal
- provide a single public dispatcher for dimension-based entry

# Notes
- This module assumes the same one-dimensional quadrature nodes and weights are
  used on every axis.
- The public entry point of this module is
  [`quadrature_threaded_subgrid`](@ref).
- Specialized implementations are provided for dimensions 1 through 4, with a
  generic fallback for higher dimensions.
"""
module QuadratureDispatchThreadedSubgrid

import Base.Threads

import ..JobLoggerTools
import ..QuadratureNodes

"""
    _chunk_range(
        n::Int,
        nparts::Int,
        part::Int,
    ) -> UnitRange{Int}

Return the index range corresponding to the `part`-th chunk when splitting
`1:n` into `nparts` contiguous pieces.

# Function description
This helper partitions a one-dimensional index interval into `nparts`
contiguous chunks using integer division, and returns the subrange assigned to
the requested chunk number `part`. The partitioning is designed so that chunk
sizes differ by at most one element.

# Arguments
- `n::Int`:
  Total number of indices.
- `nparts::Int`:
  Number of contiguous chunks.
- `part::Int`:
  One-based chunk selector.

# Returns
- `UnitRange{Int}`:
  Contiguous index range assigned to the requested chunk.

# Notes
- This helper does not itself validate that `part` lies in `1:nparts`.
- Empty ranges may occur when `nparts > n`.
"""
function _chunk_range(
    n::Int,
    nparts::Int,
    part::Int,
)
    start = fld((part - 1) * n, nparts) + 1
    stop  = fld(part * n, nparts)
    return start:stop
end

"""
    _choose_axis_splits(
        nthreads_req::Int,
        dim::Int,
        ngrid::Int,
    ) -> Vector{Int}

Choose a per-axis split configuration for threaded tensor-product subgrid
partitioning.

# Function description
This helper searches for a vector of axis split counts whose product does not
exceed the requested thread budget `nthreads_req`, while trying to maximize the
number of usable blocks and keep the split pattern balanced across dimensions.
The returned vector has length `dim`, and each entry gives the number of
subranges to create along the corresponding axis.

# Arguments
- `nthreads_req::Int`:
  Requested number of worker threads or blocks.
- `dim::Int`:
  Number of dimensions.
- `ngrid::Int`:
  Reference upper bound for the number of quadrature nodes per axis.
  In rectangular axis-wise domains, this is typically chosen as the maximum
  axis length so that a common split pattern can be constructed and later
  clipped to each axis.

# Returns
- `Vector{Int}`:
  Per-axis split counts.

# Errors
- Throws `ArgumentError` if `nthreads_req < 1`.
- Throws `ArgumentError` if `dim < 1`.
- Throws `ArgumentError` if `ngrid < 1`.

# Notes
- The returned split counts are bounded above by `min(nthreads_req, ngrid)`.
- The search favors larger block counts first, then more balanced split
  patterns.
"""
function _choose_axis_splits(
    nthreads_req::Int,
    dim::Int,
    ngrid::Int,
)
    nthreads_req >= 1 || throw(ArgumentError("nthreads_req must be ≥ 1"))
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    ngrid >= 1 || throw(ArgumentError("ngrid must be ≥ 1"))

    maxsplit = min(nthreads_req, ngrid)

    best = ones(Int, dim)
    best_prod = 1
    best_score = typemax(Int)

    cur = ones(Int, dim)

    function rec(d::Int, prod::Int)
        if d > dim
            score = maximum(cur) - minimum(cur)
            if (prod > best_prod) || (prod == best_prod && score < best_score)
                best .= cur
                best_prod = prod
                best_score = score
            end
            return
        end

        max_here = min(maxsplit, fld(nthreads_req, prod))
        for s in 1:max_here
            newprod = prod * s
            newprod > nthreads_req && break
            cur[d] = s
            rec(d + 1, newprod)
        end
    end

    rec(1, 1)
    return best
end

"""
    _block_ranges_from_splits(
        ngrid::Int,
        splits::AbstractVector{<:Integer},
    )

Construct tensor-product subgrid blocks from a per-axis split specification.

# Function description
This helper expands the one-dimensional split counts in `splits` into the full
collection of rectangular tensor-product blocks. Each returned block is a tuple
of `UnitRange{Int}` objects, one per axis, describing the index subranges to be
processed together.

# Arguments
- `ngrid::Int`:
  Number of quadrature nodes per axis.
- `splits::AbstractVector{<:Integer}`:
  Number of chunks to create along each axis.

# Returns
- `Vector{NTuple{dim,UnitRange{Int}}}`:
  Tensor-product block ranges, where `dim = length(splits)`.

# Notes
- The block count is the product of all entries in `splits`.
- Each axis range is constructed using [`_chunk_range`](@ref).
"""
function _block_ranges_from_splits(
    ngrid::Int,
    splits::AbstractVector{<:Integer},
)
    dim = length(splits)
    blocks = Vector{NTuple{dim,UnitRange{Int}}}()

    cur = Vector{UnitRange{Int}}(undef, dim)

    function rec(d::Int)
        if d > dim
            push!(blocks, Tuple(cur))
            return
        end
        for p in 1:splits[d]
            cur[d] = _chunk_range(ngrid, splits[d], p)
            rec(d + 1)
        end
    end

    rec(1)
    return blocks
end

"""
    _effective_nthreads_req(
        nthreads_req::Int,
    ) -> Int

Clamp the requested thread count to the number of available Julia threads.

# Function description
This helper validates that the requested thread count is positive, then returns
the effective thread count actually usable by the backend, namely

```julia
min(nthreads_req, Threads.nthreads())
```

# Arguments

* `nthreads_req::Int`:
  Requested number of threads.

# Returns

* `Int`:
  Effective thread count used by the threaded subgrid backend.

# Errors

* Throws `ArgumentError` if `nthreads_req < 1`.

# Notes

* This helper does not guarantee that all effective threads will receive equal
  work.
"""
@inline function _effective_nthreads_req(
    nthreads_req::Int,
)::Int
    nthreads_req >= 1 || throw(ArgumentError("nthreads_req must be ≥ 1"))
    return min(nthreads_req, Threads.nthreads())
end

include("QuadratureDispatchThreadedSubgrid/quadrature_1d_threaded_subgrid.jl")
include("QuadratureDispatchThreadedSubgrid/quadrature_2d_threaded_subgrid.jl")
include("QuadratureDispatchThreadedSubgrid/quadrature_3d_threaded_subgrid.jl")
include("QuadratureDispatchThreadedSubgrid/quadrature_4d_threaded_subgrid.jl")
include("QuadratureDispatchThreadedSubgrid/quadrature_nd_threaded_subgrid.jl")

"""
    quadrature_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        nthreads_req::Int = Threads.nthreads(),
        λ = nothing,
        real_type = nothing,
    )

Unified public dispatcher for thread-parallel subgrid tensor-product
quadrature.

# Function description
This function is the main public entry point of the threaded subgrid backend.
It dispatches to the matching dimension-specific implementation for `dim = 1`,
`2`, `3`, or `4`, and uses the generic ND fallback for higher dimensions.

Both hypercube-style scalar bounds and axis-wise rectangular bounds are
supported. Rectangular-domain support is provided by the selected backend.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `dim::Int`:
  Number of dimensions.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ = nothing`:
  Optional extra rule parameter forwarded to the node/weight generator.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- Quadrature approximation produced by the selected threaded subgrid backend,
  in the active scalar type.

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Throws `ArgumentError` indirectly if axis-wise bounds do not match the
  requested dimensionality.
- Propagates errors from the selected dimension-specific routine.

# Notes
- This dispatcher provides a single interface over both specialized and generic
  threaded subgrid implementations.
"""
function quadrature_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    nthreads_req::Int = Threads.nthreads(),
    λ = nothing,
    real_type = nothing,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    if dim == 1
        return quadrature_1d_threaded_subgrid(
            f, 
            a, 
            b, 
            N, 
            rule, 
            boundary;
            nthreads_req = nthreads_req,
            λ = λT,
            real_type = T,
        )
    elseif dim == 2
        return quadrature_2d_threaded_subgrid(
            f, 
            a, 
            b, 
            N, 
            rule, 
            boundary;
            nthreads_req = nthreads_req,
            λ = λT,
            real_type = T,
        )
    elseif dim == 3
        return quadrature_3d_threaded_subgrid(
            f, 
            a, 
            b, 
            N, 
            rule, 
            boundary;
            nthreads_req = nthreads_req,
            λ = λT,
            real_type = T,
        )
    elseif dim == 4
        return quadrature_4d_threaded_subgrid(
            f, 
            a, 
            b, 
            N, 
            rule, 
            boundary;
            nthreads_req = nthreads_req,
            λ = λT,
            real_type = T,
        )
    else
        return quadrature_nd_threaded_subgrid(
            f, 
            a, 
            b, 
            N, 
            rule, 
            boundary;
            dim = dim,
            nthreads_req = nthreads_req,
            λ = λT,
            real_type = T,
        )
    end
end

end  # module QuadratureDispatchThreadedSubgrid