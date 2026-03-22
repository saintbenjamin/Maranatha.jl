# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_clip_nd_threaded_subgrid_ranges.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

function _clip_nd_threaded_subgrid_ranges(
    ranges_g,
    lens::AbstractVector{<:Integer},
    dim::Int,
)
    ranges = Vector{UnitRange{Int}}(undef, dim)

    for d in 1:dim
        lo = max(first(ranges_g[d]), 1)
        hi = min(last(ranges_g[d]), lens[d])

        lo > hi && return nothing
        ranges[d] = lo:hi
    end

    return ranges
end
