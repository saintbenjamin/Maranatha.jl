# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_quadrature_nd_threaded_from_nodes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
