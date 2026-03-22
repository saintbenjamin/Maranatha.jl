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

        local_sum = zero(T)

        for I in Iterators.product(ranges...)
            wprod = one(T)

            for d in 1:dim
                ii = I[d]
                wprod *= ws_list[d][ii]
            end

            if !iszero(wprod)
                vals = ntuple(d -> xs_list[d][I[d]], dim)
                local_sum += wprod * f(vals...)
            end
        end

        partial[bid] = local_sum
    end

    return sum(partial)
end
