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
    _quadrature_nd_threaded_subgrid_from_nodes(
        f,
        xs_list::Vector{Vector{T}},
        ws_list::Vector{Vector{T}},
        dim::Int,
        nthreads_eff::Int,
    ) where {T}

Evaluate a generic `dim`-dimensional tensor-product quadrature approximation
from precomputed per-axis nodes and weights.

# Function description
This internal helper performs the actual generic ND tensor-product traversal
after all one-dimensional quadrature nodes and weights have already been
constructed. It is used by [`quadrature_nd_threaded_subgrid`](@ref) once domain,
rule, boundary, and scalar-type handling have been resolved.

If `nthreads_eff <= 1`, or if every axis has length one, the helper traverses
the full tensor-product grid serially with an iterative multi-index update.

Otherwise, it partitions a reference global grid into subgrid blocks, clips each
block to the valid range of every axis, evaluates each block independently on a
Julia thread, and reduces the block-local contributions at the end. Integrand
calls are issued through immutable tuples built from the active axis nodes.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `xs_list::Vector{Vector{T}}`:
  Per-axis quadrature nodes. Entry `xs_list[d]` contains the one-dimensional
  nodes for axis `d`.
- `ws_list::Vector{Vector{T}}`:
  Per-axis quadrature weights. Entry `ws_list[d]` contains the one-dimensional
  weights for axis `d`.
- `dim::Int`:
  Number of dimensions.
- `nthreads_eff::Int`:
  Effective number of Julia threads to use for the subgrid decomposition.

# Returns
- Quadrature approximation in scalar type `T`.

# Errors
- May throw indexing-related errors if `length(xs_list) != dim`,
  `length(ws_list) != dim`, or if node and weight arrays are inconsistent.
- Propagates exceptions thrown by the integrand `f`.

# Notes
- This helper does not validate rule, boundary, or domain specifications.
  Those checks belong to the caller.
- Axis lengths may differ across dimensions.
- Zero-weight tensor-product points are skipped.
- This routine is internal and not intended as the primary public entry point.
"""
Base.@noinline function _quadrature_nd_threaded_subgrid_from_nodes(
    f,
    xs_list::Vector{Vector{T}},
    ws_list::Vector{Vector{T}},
    dim::Int,
    nthreads_eff::Int,
) where {T}
    lens = [length(xs_list[d]) for d in 1:dim]

    if nthreads_eff <= 1 || all(==(1), lens)
        return _quadrature_nd_serial_from_nodes(
            f,
            xs_list,
            ws_list,
            lens,
            dim,
        )
    end

    return _quadrature_nd_threaded_from_nodes(
        f,
        xs_list,
        ws_list,
        lens,
        dim,
        nthreads_eff,
    )
end

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
This function is the generic ND threaded-subgrid backend used when no dedicated
low-dimensional specialized kernel is selected. It resolves the active scalar
type, validates the supplied rule and boundary specifications, constructs the
one-dimensional nodes and weights for each axis, and then delegates the actual
tensor-product traversal to
[`_quadrature_nd_threaded_subgrid_from_nodes`](@ref).

It supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds, and the domain becomes
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

The same shared-versus-axis-wise convention is supported for `rule` and
`boundary`.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a
  tuple/vector of per-axis lower bounds of length `dim`.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a
  tuple/vector of per-axis upper bounds of length `dim`.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature rule specification. This may be either a scalar rule symbol shared
  across all axes or a tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary specification. This may be either a scalar boundary symbol shared
  across all axes or a tuple/vector of per-axis boundary symbols of length
  `dim`.
- `dim::Int`:
  Number of dimensions.
- `nthreads_req::Int = Threads.nthreads()`:
  Requested number of threads. The effective value is normalized internally.
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
- Throws validation errors if axis-wise `rule` or `boundary` specifications are
  inconsistent with `dim`.
- Propagates node-construction and integrand-evaluation errors from the
  underlying quadrature and traversal helpers.

# Notes
- This implementation is more general than the specialized 1D–4D threaded
  kernels, but may be less efficient.
- Hypercube domains reuse the same scalar interval on every axis, but may still
  use axis-wise `rule` / `boundary` specifications.
- Rectangular domains construct nodes and weights independently for each axis.
- The actual tensor-product traversal is performed by
  [`_quadrature_nd_threaded_subgrid_from_nodes`](@ref).
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

    node_state = _build_nd_threaded_subgrid_nodes_weights(
        a,
        b,
        N,
        rule,
        boundary;
        dim = dim,
        λ = λT,
        real_type = T,
    )

    return _quadrature_nd_threaded_subgrid_from_nodes(
        f,
        node_state.xs_list,
        node_state.ws_list,
        dim,
        nthreads_eff,
    )
end
