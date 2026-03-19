# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/quadrature_2d_threaded_subgrid.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    quadrature_2d_threaded_subgrid(
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

Evaluate a two-dimensional tensor-product quadrature approximation using
thread-parallel subgrid partitioning.

# Function description
This function supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval `[a, b]` is used on both axes.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors, they are interpreted as per-axis bounds,
  and the domain becomes ``[a_1,b_1] \\times [a_2,b_2]``.

In threaded mode, the tensor-product grid is partitioned into rectangular
subblocks across both axes, and each block is processed by a Julia thread.
For rectangular domains with different axis lengths, the block decomposition is
built from a common reference grid and then clipped to the valid range of each axis.
In non-threaded mode, the full tensor-product loop is evaluated serially.

# Arguments
- `f`:
  Two-dimensional integrand callable.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across both axes, or a length-2
  tuple/vector of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across both axes, or a length-2
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
- Quadrature approximation of the two-dimensional integral in the active scalar type.

# Errors
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 2`
  or `length(b) != 2`.
- Propagates errors from `QuadratureNodes.get_quadrature_1d_nodes_weights`.
- Propagates any error thrown by `f`.

# Notes
- Hypercube domains reuse the same one-dimensional nodes and weights on both axes.
- Rectangular domains construct nodes and weights independently for each axis.
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
    λ = nothing,
    real_type = nothing,
)
    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == 2 || throw(ArgumentError("length(a) must be 2 for 2D"))
        length(b) == 2 || throw(ArgumentError("length(b) must be 2 for 2D"))
        promote_type(typeof(a[1]), typeof(a[2]), typeof(b[1]), typeof(b[2]))
    else
        promote_type(typeof(a), typeof(b))
    end
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    nthreads_eff = _effective_nthreads_req(nthreads_req)

    if !(a isa AbstractVector || a isa Tuple)
        xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
            a, b, N, rule, boundary;
            λ = λT,
            real_type = T,
        )
        ys, wy = xs, wx
    else
        xs, wx = QuadratureNodes.get_quadrature_1d_nodes_weights(
            a[1], b[1], N, rule, boundary;
            λ = λT,
            real_type = T,
        )
        ys, wy = QuadratureNodes.get_quadrature_1d_nodes_weights(
            a[2], b[2], N, rule, boundary;
            λ = λT,
            real_type = T,
        )
    end

    nx = length(xs)
    ny = length(ys)

    if nthreads_eff <= 1 || (nx == 1 && ny == 1)
        total = zero(T)
        @inbounds for i in eachindex(xs)
            xi = xs[i]
            wi = wx[i]
            for j in eachindex(ys)
                w = wi * wy[j]
                iszero(w) && continue
                total += w * f(xi, ys[j])
            end
        end
        return total
    end

    ngrid = max(nx, ny)
    splits = _choose_axis_splits(nthreads_eff, 2, ngrid)
    blocks = _block_ranges_from_splits(ngrid, splits)

    JobLoggerTools.println_benji("Global grid: $(nx)×$(ny) points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(T, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        r1g, r2g = blocks[bid]
        r1 = max(first(r1g), 1):min(last(r1g), nx)
        r2 = max(first(r2g), 1):min(last(r2g), ny)

        isempty(r1) || isempty(r2) && continue

        local_sum = zero(T)

        @inbounds for i in r1
            xi = xs[i]
            wi = wx[i]
            for j in r2
                w = wi * wy[j]
                iszero(w) && continue
                local_sum += w * f(xi, ys[j])
            end
        end

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end