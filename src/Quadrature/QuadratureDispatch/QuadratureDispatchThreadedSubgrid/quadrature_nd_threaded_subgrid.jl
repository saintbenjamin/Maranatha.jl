# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/quadrature_nd_threaded_subgrid.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
        λ = nothing,
        real_type = nothing,
    )

Evaluate a generic `dim`-dimensional tensor-product quadrature approximation
using thread-parallel subgrid partitioning.

# Function description
This function provides the generic fallback implementation for dimensions not
covered by the specialized 1D–4D kernels.

It supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds, and the domain becomes
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

In threaded mode, each rectangular subgrid block is processed independently by a
Julia thread. For rectangular domains with different axis lengths, the block
decomposition is built from a common reference grid and then clipped to each
axis length. In serial mode, the entire tensor-product grid is traversed with an
iterative multi-index update.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds of length `dim`.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds of length `dim`.
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
- `λ = nothing`:
  Optional extra rule parameter forwarded to the node/weight generator.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- Quadrature approximation of the `dim`-dimensional integral in the active scalar type.

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.

# Notes
- This implementation is more general than the specialized 1D–4D versions, but
  may be less efficient.
- Hypercube domains reuse the same one-dimensional nodes and weights on every axis.
- Rectangular domains construct nodes and weights independently for each axis.
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
    λ = nothing,
    real_type = nothing,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    nthreads_eff = _effective_nthreads_req(nthreads_req)

    if !(a isa AbstractVector || a isa Tuple)
        xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(
            a, b, N, rule, boundary;
            λ = λT,
            real_type = T,
        )
        xs_list = [xs for _ in 1:dim]
        ws_list = [ws for _ in 1:dim]
    else
        xs_list = Vector{Vector{T}}(undef, dim)
        ws_list = Vector{Vector{T}}(undef, dim)
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a[d], b[d], N, rule, boundary;
                λ = λT,
                real_type = T,
            )
        end
    end

    lens = [length(xs_list[d]) for d in 1:dim]

    if nthreads_eff <= 1 || all(==(1), lens)
        idx = ones(Int, dim)
        args = Vector{T}(undef, dim)
        total = zero(T)

        @inbounds while true
            wprod = one(T)
            for d in 1:dim
                ii = idx[d]
                args[d] = xs_list[d][ii]
                wprod *= ws_list[d][ii]
            end

            iszero(wprod) || (total += wprod * f(args...))

            d = dim
            while d >= 1
                idx[d] += 1
                if idx[d] <= lens[d]
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

    ngrid = maximum(lens)
    splits = _choose_axis_splits(nthreads_eff, dim, ngrid)
    blocks = _block_ranges_from_splits(ngrid, splits)

    JobLoggerTools.println_benji("Global grid: $(join(lens, '×')) points | threads: $(nthreads_eff) → axis splits = $(splits) → total subgrids = $(length(blocks))")

    partial = zeros(T, Threads.maxthreadid())

    Threads.@threads for bid in eachindex(blocks)
        ranges_g = blocks[bid]

        ranges = ntuple(d -> begin
            lo = max(first(ranges_g[d]), 1)
            hi = min(last(ranges_g[d]), lens[d])
            lo:hi
        end, dim)

        any(isempty, ranges) && continue

        args = Vector{T}(undef, dim)
        local_sum = zero(T)

        function _walk_block(d::Int, wprod)
            if d > dim
                local_sum += wprod * f(args...)
                return
            end

            xd = xs_list[d]
            wd = ws_list[d]

            @inbounds for ii in ranges[d]
                wi = wd[ii]
                iszero(wi) && continue
                args[d] = xd[ii]
                _walk_block(d + 1, wprod * wi)
            end
        end

        _walk_block(1, one(T))

        partial[Threads.threadid()] += local_sum
    end

    return sum(partial)
end