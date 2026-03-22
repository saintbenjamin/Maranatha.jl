# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchCUDA/internal/_prepare_cuda_launch_state.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _prepare_cuda_launch_state(
        xs_list,
        ws_list,
        threads::Int,
    ) -> NamedTuple

Pack host-side axis nodes and weights into CUDA launch-ready data structures.

# Function description
This helper converts per-axis host-side node and weight vectors into the dense
matrix representation used by [`quadrature_cuda`](@ref), transfers those
matrices to device memory, and prepares the auxiliary launch metadata needed by
the CUDA kernel.

# Arguments
- `xs_list`:
  Host-side per-axis node vectors.
- `ws_list`:
  Host-side per-axis weight vectors.
- `threads::Int`:
  CUDA threads per block.

# Returns
- `NamedTuple`:
  A bundle with fields:
  - `xs_d`
  - `ws_d`
  - `lens_vec`
  - `lens`
  - `total_points`
  - `blocks`
  - `out`

# Errors
- Propagates allocation and device-transfer errors from `CUDA`.

# Notes
- This helper performs no kernel launch and no final reduction.
"""
function _prepare_cuda_launch_state(
    xs_list,
    ws_list,
    threads::Int,
)
    T = eltype(xs_list[1])

    lens_vec = [length(xs_list[d]) for d in eachindex(xs_list)]
    maxn = maximum(lens_vec)
    dim = length(xs_list)

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

    return (;
        xs_d = xs_d,
        ws_d = ws_d,
        lens_vec = lens_vec,
        lens = lens,
        total_points = total_points,
        blocks = blocks,
        out = out,
    )
end
