# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/quadrature_1d_threaded_subgrid.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_1d_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        nthreads_req::Int = Threads.nthreads(),
        λ = nothing,
        real_type = nothing,
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
  Quadrature rule specification valid for `dim = 1`.
- `boundary`:
  Boundary specification valid for `dim = 1`.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ = nothing`:
  Optional extra rule parameter forwarded to the node/weight generator.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- Quadrature approximation of the one-dimensional integral in the active scalar type.

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
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    nthreads_eff = _effective_nthreads_req(nthreads_req)

    b1 = QuadratureBoundarySpec._boundary_at(boundary, 1, 1)

    xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a,
        b,
        N,
        rule,
        b1;
        λ = λT,
        real_type = T,
    )
    nx = length(xs)

    if nthreads_eff <= 1 || nx == 1
        total = zero(T)
        @inbounds for i in eachindex(xs)
            w = wx[i]
            iszero(w) && continue
            total += w * f(xs[i])
        end
        return total
    end

    splits = _choose_axis_splits(nthreads_eff, 1, nx)
    blocks = _block_ranges_from_splits(nx, splits)

    JobLoggerTools.println_benji(
        "Global grid: $(nx)^1 points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))"
    )

    partial = zeros(T, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        (r1,) = blocks[bid]
        local_sum = zero(T)

        @inbounds for i in r1
            w = wx[i]
            iszero(w) && continue
            local_sum += w * f(xs[i])
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end
