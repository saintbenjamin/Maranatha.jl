# ============================================================================
# src/ErrorEstimate/ErrorGaussRefinement.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorGaussRefinement

import ..JobLoggerTools
import ..Gauss
import ..QuadratureUtils
import ..QuadratureDispatch

"""
    _require_gauss_rule(
        rule::Symbol
    ) -> Nothing

Validate that `rule` belongs to the supported Gauss-family quadrature rules.

# Function description
This internal helper checks whether the supplied quadrature-rule symbol is a
Gauss-family rule recognized by the quadrature layer. It is used as a guard
before calling Gauss-specific refinement routines.

# Arguments
- `rule::Symbol`:
  Quadrature rule symbol to validate.

# Returns
- `nothing`

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a supported
  Gauss-family rule.

# Notes
- This helper validates only the rule family.
- It does not validate `boundary`, `N`, or `dim`.
"""
@inline function _require_gauss_rule(
    rule::Symbol
)::Nothing
    Gauss._is_gauss_rule(rule) ||
        JobLoggerTools.error_benji(
            "ErrorGaussRefinement only supports Gauss-family rules (got rule=$rule)"
        )
    return nothing
end

"""
    _require_gauss_inputs(
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol,
    ) -> Nothing

Validate the basic inputs required by the Gauss refinement estimator.

# Function description
This helper performs the common input checks used by the Gauss-family
refinement-based error-estimation layer. It verifies that the subdivision count
and dimensionality are valid, confirms that `rule` belongs to the Gauss family,
and delegates boundary validation to `QuadratureUtils._decode_boundary`.

# Arguments
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Problem dimensionality.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- `nothing`

# Errors
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Throws if `rule` is not a supported Gauss-family rule.
- Throws if `boundary` is invalid.

# Notes
- This helper centralizes the shared validation logic for the public and
  internal Gauss refinement routines.
"""
@inline function _require_gauss_inputs(
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol,
)::Nothing
    (N >= 1)   || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")
    _require_gauss_rule(rule)
    QuadratureUtils._decode_boundary(boundary)
    return nothing
end

"""
    _quadrature_value_gauss(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        threaded_subgrid::Bool = false,
        real_type = nothing,
    ) -> Real

Evaluate the Gauss-family quadrature approximation of `f`.

# Function description
This helper validates the input configuration and then calls
`QuadratureDispatch.quadrature` to compute the quadrature approximation using
the requested Gauss-family rule, boundary condition, subdivision count, and
dimensionality.

Two domain conventions are supported:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds, and the domain becomes
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

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
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the quadrature backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and quadrature
  evaluation.

# Returns
- `Real`:
  The quadrature value produced by the Gauss-family backend, in the active scalar type.

# Errors
- Propagates validation errors from [`_require_gauss_inputs`](@ref).
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Propagates errors from `QuadratureDispatch.quadrature`.

# Notes
- This helper performs no derivative-based work.
- It is used internally by the refinement-based Gauss error estimator.
"""
@inline function _quadrature_value_gauss(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end

    _require_gauss_inputs(N, dim, rule, boundary)

    q = QuadratureDispatch.quadrature(
        f,
        a isa AbstractVector || a isa Tuple ? map(x -> convert(T, x), a) : convert(T, a),
        b isa AbstractVector || b isa Tuple ? map(x -> convert(T, x), b) : convert(T, b),
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    return q
end

"""
    _estimate_by_refinement_gauss(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        threaded_subgrid::Bool = false,
        real_type = nothing,
    )

Estimate the Gauss-family quadrature error by comparing coarse and refined
quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
Gauss-family quadrature rules. It computes

- a coarse quadrature value using `N` subdivisions, and
- a refined quadrature value using `2N` subdivisions,

then forms the refinement difference

```julia
diff = q_fine - q_coarse
```

Two domain conventions are supported:

* **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the mesh sizes are scalar quantities.

* **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, the mesh sizes are
  constructed componentwise and stored as per-axis tuples.

The absolute value of this difference is used as the effective error estimate,
while the returned named tuple also records the signed difference, mesh sizes,
and both quadrature evaluations.

# Arguments

* `f`:
  Integrand callable accepting `dim` positional arguments.
* `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds of length `dim`.
* `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds of length `dim`.
* `N::Int`:
  Coarse subdivision count.
* `dim::Int`:
  Number of dimensions.
* `rule::Symbol`:
  Gauss-family quadrature rule symbol.
* `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments

* `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the coarse and refined
  quadrature calls.
* `real_type = nothing`:
  Optional scalar type used internally for bound conversion, mesh sizes,
  and quadrature evaluation.

# Returns

* `NamedTuple` with fields:

  * `method`      : method tag `:gauss_refinement_difference`
  * `rule`        : quadrature rule symbol
  * `boundary`    : boundary-condition symbol
  * `N_coarse`    : coarse subdivision count
  * `N_fine`      : refined subdivision count (`2N`)
  * `dim`         : dimensionality
  * `h_coarse`    : coarse mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `h_fine`      : refined mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `q_coarse`    : coarse quadrature value
  * `q_fine`      : refined quadrature value
  * `estimate`    : absolute refinement difference
  * `signed_diff` : signed refinement difference
  * `reference`   : refined quadrature value used as the internal reference

# Errors

* Propagates validation errors from [`_require_gauss_inputs`](@ref).
* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
* Propagates errors from the quadrature-evaluation layer.

# Notes

* This estimator does not use derivatives, jets, or residual moments.
* The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
"""
function _estimate_by_refinement_gauss(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end

    _require_gauss_inputs(N, dim, rule, boundary)

    if a isa AbstractVector || a isa Tuple
        aa = ntuple(i -> convert(T, a[i]), dim)
        bb = ntuple(i -> convert(T, b[i]), dim)

        h_coarse = ntuple(i -> (bb[i] - aa[i]) / T(N), dim)
        h_fine   = ntuple(i -> (bb[i] - aa[i]) / T(2N), dim)
    else
        aa = convert(T, a)
        bb = convert(T, b)

        h_coarse = (bb - aa) / T(N)
        h_fine   = (bb - aa) / T(2N)
    end

    q_coarse = _quadrature_value_gauss(
        f,
        aa,
        bb,
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    q_fine = _quadrature_value_gauss(
        f,
        aa,
        bb,
        2N,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    diff = q_fine - q_coarse

    return (;
        method      = :gauss_refinement_difference,
        rule        = rule,
        boundary    = boundary,
        N_coarse    = N,
        N_fine      = 2N,
        dim         = dim,
        h_coarse    = h_coarse,
        h_fine      = h_fine,
        q_coarse    = q_coarse,
        q_fine      = q_fine,
        estimate    = abs(diff),
        signed_diff = diff,
        reference   = q_fine,
    )
end

"""
    error_estimate_refinement_gauss(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid::Bool = false,
        real_type = nothing,
    )

Unified public dispatcher for Gauss-family refinement-based error estimation.

# Function description
This function provides the main Gauss-family-specific entry point for the
refinement-based error-estimation layer.

It supports both of the following domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds.

The routine validates the rule family and boundary selector, then dispatches to
[`_estimate_by_refinement_gauss`](@ref).

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds.
- `N`:
  Coarse subdivision count.
- `dim`:
  Number of dimensions.
- `rule`:
  Gauss-family quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the underlying refinement calls.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and refinement
  evaluation.

# Returns
- The named tuple produced by the selected refinement estimator.

# Errors
- Throws if `rule` is not a supported Gauss-family rule.
- Throws if `boundary` is invalid.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Propagates errors from the refinement-estimation routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on derivative backends or jet construction.
"""
function error_estimate_refinement_gauss(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    _require_gauss_rule(rule)
    QuadratureUtils._decode_boundary(boundary)

    return _estimate_by_refinement_gauss(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = real_type,
    )
end

end  # module ErrorGaussRefinement