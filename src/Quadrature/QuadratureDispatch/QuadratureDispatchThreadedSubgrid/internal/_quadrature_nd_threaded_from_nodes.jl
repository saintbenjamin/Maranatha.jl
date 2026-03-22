# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_quadrature_nd_threaded_from_nodes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _quadrature_nd_threaded_from_nodes(
        f,
        xs_list::Vector{Vector{T}},
        ws_list::Vector{Vector{T}},
        lens::Vector{Int},
        dim::Int,
        nthreads_eff::Int,
    ) where {T} -> T

Evaluate a generic ND tensor-product quadrature sum using threaded subgrid
decomposition from precomputed per-axis nodes and weights.

# Function description
This helper partitions a common reference grid into rectangular blocks,
clips each block to the valid per-axis node ranges, and assigns block-local
tensor-product traversal to Julia threads.

Within each block it uses an explicit odometer loop over local indices,
multiplies the active per-axis weights, evaluates the integrand at the current
coordinates, and accumulates a block-local scalar sum. The final result is the
sum of all block-local partials.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `xs_list::Vector{Vector{T}}`:
  Per-axis quadrature nodes.
- `ws_list::Vector{Vector{T}}`:
  Per-axis quadrature weights.
- `lens::Vector{Int}`:
  Valid node counts on each axis.
- `dim::Int`:
  Problem dimensionality.
- `nthreads_eff::Int`:
  Effective number of CPU threads requested for block decomposition.

# Returns
- `T`:
  Threaded tensor-product quadrature approximation in the active scalar type.

# Errors
- May propagate indexing errors if node, weight, and length arrays are
  inconsistent.
- Propagates exceptions thrown by `f`.

# Notes
- Reduction is performed through block-local partial sums indexed by block id.
- Zero-weight tensor-product nodes are skipped.
- This helper assumes that node and weight construction has already been
  completed by the caller.
"""
function _quadrature_nd_threaded_from_nodes(
    f,
    xs_list::Vector{Vector{T}},
    ws_list::Vector{Vector{T}},
    lens::Vector{Int},
    dim::Int,
    nthreads_eff::Int,
) where {T}
    ngrid = maximum(lens)
    splits = _choose_axis_splits(nthreads_eff, dim, ngrid)
    blocks = _block_ranges_from_splits(ngrid, splits)

    partial = zeros(T, length(blocks))

    Threads.@threads for bid in eachindex(blocks)
        ranges = _clip_nd_threaded_subgrid_ranges(
            blocks[bid],
            lens,
            dim,
        )

        isnothing(ranges) && continue

        starts = Vector{Int}(undef, dim)
        stops = Vector{Int}(undef, dim)
        idx = Vector{Int}(undef, dim)
        args = Vector{T}(undef, dim)

        @inbounds for d in 1:dim
            starts[d] = first(ranges[d])
            stops[d] = last(ranges[d])
            idx[d] = starts[d]
        end

        local_sum = zero(T)

        while true
            wprod = one(T)

            @inbounds for d in 1:dim
                ii = idx[d]
                args[d] = xs_list[d][ii]
                wprod *= ws_list[d][ii]
            end

            if !iszero(wprod)
                local_sum += wprod * f(args...)
            end

            carry_dim = dim
            while carry_dim >= 1
                idx[carry_dim] += 1
                if idx[carry_dim] <= stops[carry_dim]
                    break
                else
                    idx[carry_dim] = starts[carry_dim]
                    carry_dim -= 1
                end
            end

            carry_dim == 0 && break
        end

        partial[bid] = local_sum
    end

    return sum(partial)
end
