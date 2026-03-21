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
  Either a scalar rule symbol shared across both axes, or a length-2
  tuple/vector of per-axis rule symbols.
- `boundary`:
  Either a scalar boundary symbol shared across both axes, or a length-2
  tuple/vector of per-axis boundary symbols.
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
    return quadrature_nd_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim = 2,
        nthreads_req = nthreads_req,
        λ = λ,
        real_type = real_type,
    )
end
