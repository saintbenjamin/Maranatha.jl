# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_clip_nd_threaded_subgrid_ranges.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _clip_nd_threaded_subgrid_ranges(
        ranges_g,
        lens::AbstractVector{<:Integer},
        dim::Int,
    ) -> Union{Vector{UnitRange{Int}}, Nothing}

Clip a reference ND subgrid block to the valid axis ranges of the active
tensor-product grid.

# Function description
This helper takes a block range tuple produced from a common reference grid and
intersects it with the valid axis lengths stored in `lens`.

If every axis has a nonempty intersection, the clipped per-axis ranges are
returned. If any axis clips to an empty interval, the helper returns `nothing`
to signal that the block can be skipped.

# Arguments
- `ranges_g`:
  Reference-grid block ranges, one range per axis.
- `lens::AbstractVector{<:Integer}`:
  Valid node counts on each axis.
- `dim::Int`:
  Problem dimensionality.

# Returns
- `Vector{UnitRange{Int}}`:
  Clipped per-axis ranges when the block is valid.
- `nothing`:
  Returned when at least one clipped axis range is empty.

# Errors
- May propagate indexing errors if `ranges_g` or `lens` are inconsistent with
  `dim`.

# Notes
- This helper is used by the threaded ND traversal to discard out-of-bounds
  reference blocks before entering the hot loop.
"""
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
