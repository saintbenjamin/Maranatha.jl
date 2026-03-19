# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/quadrature_4d_threaded_subgrid.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_4d_threaded_subgrid(
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

Evaluate a four-dimensional tensor-product quadrature approximation using
thread-parallel subgrid partitioning.

# Function description
This function supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval `[a, b]` is used on all four axes.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors, they are interpreted as per-axis bounds,
  and the domain becomes
  ``[a_1,b_1] \\times [a_2,b_2] \\times [a_3,b_3] \\times [a_4,b_4]``.

In threaded mode, the tensor-product grid is partitioned into rectangular 4D
subblocks, and each block is accumulated independently by a Julia thread.
For rectangular domains with different axis lengths, the block decomposition is
built from a common reference grid and then clipped to the valid range of each axis.
In serial mode, the full nested loop is evaluated directly.

# Arguments
- `f`:
  Four-dimensional integrand callable.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a length-4
  tuple/vector of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a length-4
  tuple/vector of per-axis upper bounds.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads.
- `λ = nothing`:
  Optional extra rule parameter forwarded to the node/weight generator.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- Quadrature approximation of the four-dimensional integral in the active scalar type.

# Errors
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 4`
  or `length(b) != 4`.
- Propagates errors from `QuadratureNodes.get_quadrature_1d_nodes_weights`.
- Propagates any error thrown by `f`.

# Notes
- Hypercube domains reuse the same one-dimensional nodes and weights on all axes.
- Rectangular domains construct nodes and weights independently for each axis.
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
    λ = nothing,
    real_type = nothing,
)
    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == 4 || throw(ArgumentError("length(a) must be 4 for 4D"))
        length(b) == 4 || throw(ArgumentError("length(b) must be 4 for 4D"))
        promote_type(typeof(a[1]), typeof(a[2]), typeof(a[3]), typeof(a[4]),
                     typeof(b[1]), typeof(b[2]), typeof(b[3]), typeof(b[4]))
    else
        promote_type(typeof(a), typeof(b))
    end
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    nthreads_eff = _effective_nthreads_req(nthreads_req)

    if !(a isa AbstractVector || a isa Tuple)
        xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary; λ = λT, real_type = T)
        ys, wy = xs, wx
        zs, wz = xs, wx
        ts, wt = xs, wx
    else
        xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(a[1], b[1], N, rule, boundary; λ = λT, real_type = T)
        ys, wy = QuadratureNodes.get_quadrature_1d_nodes_weights(a[2], b[2], N, rule, boundary; λ = λT, real_type = T)
        zs, wz = QuadratureNodes.get_quadrature_1d_nodes_weights(a[3], b[3], N, rule, boundary; λ = λT, real_type = T)
        ts, wt = QuadratureNodes.get_quadrature_1d_nodes_weights(a[4], b[4], N, rule, boundary; λ = λT, real_type = T)
    end

    nx = length(xs)
    ny = length(ys)
    nz = length(zs)
    nt = length(ts)

    if nthreads_eff <= 1 || (nx == 1 && ny == 1 && nz == 1 && nt == 1)
        total = zero(T)
        @inbounds for i in eachindex(xs)
            xi = xs[i]
            wi = wx[i]
            for j in eachindex(ys)
                yj = ys[j]
                wij = wi * wy[j]
                for k in eachindex(zs)
                    zk = zs[k]
                    wijk = wij * wz[k]
                    for l in eachindex(ts)
                        w = wijk * wt[l]
                        iszero(w) && continue
                        total += w * f(xi, yj, zk, ts[l])
                    end
                end
            end
        end
        return total
    end

    ngrid = max(nx, ny, nz, nt)
    splits = _choose_axis_splits(nthreads_eff, 4, ngrid)
    blocks = _block_ranges_from_splits(ngrid, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)×$(ny)×$(nz)×$(nt) points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(T, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        r1g, r2g, r3g, r4g = blocks[bid]
        r1 = max(first(r1g), 1):min(last(r1g), nx)
        r2 = max(first(r2g), 1):min(last(r2g), ny)
        r3 = max(first(r3g), 1):min(last(r3g), nz)
        r4 = max(first(r4g), 1):min(last(r4g), nt)

        (isempty(r1) || isempty(r2) || isempty(r3) || isempty(r4)) && continue

        local_sum = zero(T)

        @inbounds for i in r1
            xi = xs[i]
            wi = wx[i]
            for j in r2
                yj = ys[j]
                wij = wi * wy[j]
                for k in r3
                    zk = zs[k]
                    wijk = wij * wz[k]
                    for l in r4
                        w = wijk * wt[l]
                        iszero(w) && continue
                        local_sum += w * f(xi, yj, zk, ts[l])
                    end
                end
            end
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end