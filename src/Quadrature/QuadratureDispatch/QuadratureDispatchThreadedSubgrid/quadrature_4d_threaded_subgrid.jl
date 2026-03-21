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
  Either a scalar rule symbol shared across all axes, or a length-4
  tuple/vector of per-axis rule symbols.
- `boundary`:
  Either a scalar boundary symbol shared across all axes, or a length-4
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
    return quadrature_nd_threaded_subgrid(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim = 4,
        nthreads_req = nthreads_req,
        λ = λ,
        real_type = real_type,
    )
end
