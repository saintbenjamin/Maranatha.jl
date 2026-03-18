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
  Number of quadrature nodes per axis.

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

"""
    quadrature_1d_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        nthreads_req::Int = Threads.nthreads(),
        λ::Float64 = 0.0,
    )

Evaluate a one-dimensional quadrature approximation using thread-parallel
subgrid partitioning.

# Function description
This function computes the one-dimensional quadrature sum for `f` on `[a, b]`
using the requested quadrature rule and boundary selector. If threading is
beneficial, the one-dimensional node index range is partitioned into chunks and
distributed across Julia threads; otherwise, a serial fallback is used.

# Arguments
- `f`:
  One-dimensional integrand callable.
- `a`:
  Lower integration bound.
- `b`:
  Upper integration bound.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation of the one-dimensional integral.

# Notes
- If `nthreads_req <= 1` or only one quadrature node is present, the function
  uses a serial loop.
- Zero-weight nodes are skipped.
"""
function quadrature_1d_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    nthreads_req::Int = Threads.nthreads(),
    λ::Float64 = 0.0
)
    nthreads_eff = _effective_nthreads_req(nthreads_req)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        total = 0.0
        @inbounds for i in eachindex(xs)
            w = wx[i]
            iszero(w) && continue
            total += w * f(xs[i])
        end
        return total
    end

    splits = _choose_axis_splits(nthreads_eff, 1, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)^1 points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(Float64, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        (r1,) = blocks[bid]
        local_sum = 0.0

        @inbounds for i in r1
            w = wx[i]
            iszero(w) && continue
            local_sum += w * f(xs[i])
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end

"""
    quadrature_2d_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        nthreads_req::Int = Threads.nthreads(),
        λ::Float64 = 0.0,
    )

Evaluate a two-dimensional tensor-product quadrature approximation using
thread-parallel subgrid partitioning.

# Function description
This function computes the two-dimensional quadrature sum for `f(x, y)` on
`[a, b]^2`. In threaded mode, the tensor-product grid is partitioned into
rectangular subblocks across both axes, and each block is processed by a Julia
thread. In non-threaded mode, the full tensor-product loop is evaluated
serially.

# Arguments
- `f`:
  Two-dimensional integrand callable.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation of the two-dimensional integral.

# Notes
- The same one-dimensional nodes and weights are reused on both axes.
- Zero-weight tensor-product points are skipped.
"""
function quadrature_2d_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    nthreads_req::Int = Threads.nthreads(),
    λ::Float64 = 0.0
)
    nthreads_eff = _effective_nthreads_req(nthreads_req)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        total = 0.0
        @inbounds for i in eachindex(xs)
            xi = xs[i]
            wi = wx[i]
            for j in eachindex(xs)
                w = wi * wx[j]
                iszero(w) && continue
                total += w * f(xi, xs[j])
            end
        end
        return total
    end

    splits = _choose_axis_splits(nthreads_eff, 2, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)^2 points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(Float64, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        r1, r2 = blocks[bid]
        local_sum = 0.0

        @inbounds for i in r1
            xi = xs[i]
            wi = wx[i]
            for j in r2
                w = wi * wx[j]
                iszero(w) && continue
                local_sum += w * f(xi, xs[j])
            end
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end

"""
    quadrature_3d_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        nthreads_req::Int = Threads.nthreads(),
        λ::Float64 = 0.0,
    )

Evaluate a three-dimensional tensor-product quadrature approximation using
thread-parallel subgrid partitioning.

# Function description
This function computes the three-dimensional quadrature sum for `f(x, y, z)` on
`[a, b]^3`. When threading is enabled, the quadrature grid is split into
rectangular 3D subblocks, each of which is processed independently by a Julia
thread. Otherwise, the full tensor-product loop is evaluated serially.

# Arguments
- `f`:
  Three-dimensional integrand callable.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation of the three-dimensional integral.

# Notes
- The same one-dimensional nodes and weights are reused on all axes.
- Zero-weight tensor-product points are skipped.
"""
function quadrature_3d_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    nthreads_req::Int = Threads.nthreads(),
    λ::Float64 = 0.0
)
    nthreads_eff = _effective_nthreads_req(nthreads_req)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        total = 0.0
        @inbounds for i in eachindex(xs)
            xi = xs[i]
            wi = wx[i]
            for j in eachindex(xs)
                yj = xs[j]
                wij = wi * wx[j]
                for k in eachindex(xs)
                    w = wij * wx[k]
                    iszero(w) && continue
                    total += w * f(xi, yj, xs[k])
                end
            end
        end
        return total
    end

    splits = _choose_axis_splits(nthreads_eff, 3, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)^3 points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(Float64, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        r1, r2, r3 = blocks[bid]
        local_sum = 0.0

        @inbounds for i in r1
            xi = xs[i]
            wi = wx[i]
            for j in r2
                yj = xs[j]
                wij = wi * wx[j]
                for k in r3
                    w = wij * wx[k]
                    iszero(w) && continue
                    local_sum += w * f(xi, yj, xs[k])
                end
            end
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end

"""
    quadrature_4d_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        nthreads_req::Int = Threads.nthreads(),
        λ::Float64 = 0.0,
    )

Evaluate a four-dimensional tensor-product quadrature approximation using
thread-parallel subgrid partitioning.

# Function description
This function computes the four-dimensional quadrature sum for
`f(x, y, z, t)` on `[a, b]^4`. In threaded mode, the tensor-product grid is
partitioned into rectangular 4D subblocks, and each block is accumulated
independently by a Julia thread. In serial mode, the full nested loop is
evaluated directly.

# Arguments
- `f`:
  Four-dimensional integrand callable.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation of the four-dimensional integral.

# Notes
- The same one-dimensional nodes and weights are reused on all axes.
- Zero-weight tensor-product points are skipped.
"""
function quadrature_4d_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    nthreads_req::Int = Threads.nthreads(),
    λ::Float64 = 0.0
)
    nthreads_eff = _effective_nthreads_req(nthreads_req)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        total = 0.0
        @inbounds for i in eachindex(xs)
            xi = xs[i]
            wi = wx[i]
            for j in eachindex(xs)
                yj = xs[j]
                wij = wi * wx[j]
                for k in eachindex(xs)
                    zk = xs[k]
                    wijk = wij * wx[k]
                    for l in eachindex(xs)
                        w = wijk * wx[l]
                        iszero(w) && continue
                        total += w * f(xi, yj, zk, xs[l])
                    end
                end
            end
        end
        return total
    end

    splits = _choose_axis_splits(nthreads_eff, 4, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)^4 points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(Float64, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        r1, r2, r3, r4 = blocks[bid]
        local_sum = 0.0

        @inbounds for i in r1
            xi = xs[i]
            wi = wx[i]
            for j in r2
                yj = xs[j]
                wij = wi * wx[j]
                for k in r3
                    zk = xs[k]
                    wijk = wij * wx[k]
                    for l in r4
                        w = wijk * wx[l]
                        iszero(w) && continue
                        local_sum += w * f(xi, yj, zk, xs[l])
                    end
                end
            end
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end

"""
    quadrature_nd_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        nthreads_req::Int = Threads.nthreads(),
        λ::Float64 = 0.0,
    )

Evaluate a generic `dim`-dimensional tensor-product quadrature approximation
using thread-parallel subgrid partitioning.

# Function description
This function provides the generic fallback implementation for dimensions not
covered by the specialized 1D–4D kernels. It traverses the tensor-product grid
using explicit index vectors and rectangular subgrid blocks. In threaded mode,
each block is processed independently by a Julia thread. In serial mode, the
entire tensor-product grid is traversed with an iterative multi-index update.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
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
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation of the `dim`-dimensional integral.

# Errors
- Throws `ArgumentError` if `dim < 1`.

# Notes
- This implementation is more general than the specialized 1D–4D versions, but
  may be less efficient.
- The same one-dimensional nodes and weights are reused on every axis.
- Zero-weight tensor-product points are skipped.
"""
function quadrature_nd_threaded_subgrid(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    nthreads_req::Int = Threads.nthreads(),
    λ::Float64 = 0.0
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    nthreads_eff = _effective_nthreads_req(nthreads_req)

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ=λ)
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        idx = ones(Int, dim)
        args = Vector{Float64}(undef, dim)
        total = 0.0

        @inbounds while true
            wprod = 1.0
            for d in 1:dim
                ii = idx[d]
                args[d] = xs[ii]
                wprod *= ws[ii]
            end

            iszero(wprod) || (total += wprod * f(args...))

            d = dim
            while d >= 1
                idx[d] += 1
                if idx[d] <= nx
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

    splits = _choose_axis_splits(nthreads_eff, dim, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)^$(dim) points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(Float64, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        ranges = blocks[bid]

        idx = [first(r) for r in ranges]
        stop = [last(r) for r in ranges]

        args = Vector{Float64}(undef, dim)
        local_sum = 0.0

        @inbounds while true
            wprod = 1.0
            for d in 1:dim
                ii = idx[d]
                args[d] = xs[ii]
                wprod *= ws[ii]
            end

            iszero(wprod) || (local_sum += wprod * f(args...))

            d = dim
            while d >= 1
                idx[d] += 1
                if idx[d] <= stop[d]
                    break
                else
                    idx[d] = first(ranges[d])
                    d -= 1
                end
            end
            d == 0 && break
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end

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
        λ::Float64 = 0.0,
    )

Unified public dispatcher for thread-parallel subgrid tensor-product
quadrature.

# Function description
This function is the main public entry point of the threaded subgrid backend.
It dispatches to the matching dimension-specific implementation for `dim = 1`,
`2`, `3`, or `4`, and uses the generic ND fallback for higher dimensions.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound on each axis.
- `b`:
  Upper integration bound on each axis.
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
- `λ::Float64 = 0.0`:
  Optional extra rule parameter forwarded to the node/weight generator.

# Returns
- Quadrature approximation produced by the selected threaded subgrid backend.

# Errors
- Throws `ArgumentError` if `dim < 1`.
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
    λ::Float64 = 0.0
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    if dim == 1
        return quadrature_1d_threaded_subgrid(
            f, a, b, N, rule, boundary;
            nthreads_req = nthreads_req,
            λ=λ
        )
    elseif dim == 2
        return quadrature_2d_threaded_subgrid(
            f, a, b, N, rule, boundary;
            nthreads_req = nthreads_req,
            λ=λ
        )
    elseif dim == 3
        return quadrature_3d_threaded_subgrid(
            f, a, b, N, rule, boundary;
            nthreads_req = nthreads_req,
            λ=λ
        )
    elseif dim == 4
        return quadrature_4d_threaded_subgrid(
            f, a, b, N, rule, boundary;
            nthreads_req = nthreads_req,
            λ=λ
        )
    else
        return quadrature_nd_threaded_subgrid(
            f, a, b, N, rule, boundary;
            dim = dim,
            nthreads_req = nthreads_req,
            λ=λ
        )
    end
end

end  # module QuadratureDispatchThreadedSubgrid