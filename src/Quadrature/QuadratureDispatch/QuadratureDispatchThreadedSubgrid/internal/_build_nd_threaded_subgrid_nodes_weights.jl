# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_build_nd_threaded_subgrid_nodes_weights.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_nd_threaded_subgrid_nodes_weights(
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        λ,
        real_type,
    ) -> NamedTuple

Construct per-axis quadrature nodes and weights for the generic ND
threaded-subgrid backend.

# Function description
This helper centralizes the host-side preparation of one-dimensional nodes and
weights used by [`quadrature_nd_threaded_subgrid`](@ref).

It supports both scalar hypercube-style bounds and axis-wise tuple/vector
bounds. The resulting per-axis node and weight vectors are returned without
performing any tensor-product traversal.

# Arguments
- `a`, `b`:
  Integration-bound specifications.
- `N`:
  Quadrature subdivision or rule-resolution parameter.
- `rule`:
  Quadrature-rule specification.
- `boundary`:
  Boundary specification.

# Keyword arguments
- `dim::Int`:
  Problem dimensionality.
- `λ`:
  Normalized optional rule parameter.
- `real_type`:
  Active scalar type used for node and weight construction.

# Returns
- `NamedTuple`:
  A bundle with fields:
  - `xs_list`
  - `ws_list`

# Errors
- Propagates validation and node-construction errors from
  [`QuadratureBoundarySpec._validate_boundary_spec`](@ref),
  [`QuadratureRuleSpec._validate_rule_spec`](@ref), and
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).

# Notes
- This helper performs no block decomposition and no quadrature accumulation.
"""
function _build_nd_threaded_subgrid_nodes_weights(
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    λ,
    real_type,
)
    T = real_type

    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    xs_list = Vector{Vector{T}}(undef, dim)
    ws_list = Vector{Vector{T}}(undef, dim)

    if !(a isa AbstractVector || a isa Tuple)
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a,
                b,
                N,
                rule,
                boundary;
                λ = λ,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    else
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a[d],
                b[d],
                N,
                rule,
                boundary;
                λ = λ,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    end

    return (;
        xs_list = xs_list,
        ws_list = ws_list,
    )
end
