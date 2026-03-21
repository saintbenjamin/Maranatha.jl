# ============================================================================
# src/ErrorEstimate/ErrorBSplineRefinement.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorBSplineRefinement

Refinement-based error-estimation backend for B-spline quadrature rules.

# Module description
`ErrorBSplineRefinement` implements the B-spline-family branch of the
refinement-based error estimator.

It validates B-spline rule specifications, evaluates coarse and refined
quadrature values, and packages the resulting quadrature-difference estimate in
a uniform result structure understood by the higher-level error-dispatch layer.

# Notes
- This is an internal module.
- Axis-wise `rule` specifications are supported here when all axes remain in
  the B-spline family.
"""
module ErrorBSplineRefinement

import ..JobLoggerTools
import ..BSpline
import ..QuadratureDispatch
import ..QuadratureRuleSpec
import ..QuadratureBoundarySpec

"""
    _require_bspline_rule(
        rule,
        dim::Int = 1,
    ) -> Nothing

Validate that `rule` belongs to the supported B-spline quadrature family.

# Function description
This internal helper checks whether the supplied quadrature-rule specification
belongs to the B-spline family recognized by the quadrature layer. It is used
as a guard before calling B-spline-specific parsing or node/weight
construction routines.

# Arguments
- `rule`:
  Quadrature rule specification to validate. This may be either a scalar rule
  symbol or an axis-wise tuple/vector of rule symbols.
- `dim::Int = 1`:
  Problem dimensionality used when validating axis-wise rule specifications.

# Returns
- `nothing`

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a supported
  B-spline rule.

# Notes
- This helper performs only rule-family validation.
- It does not validate `boundary`, `N`, `dim`, or the smoothing parameter `λ`.
- Axis-wise rule specifications must use B-spline rules on every axis.
"""
@inline function _require_bspline_rule(
    rule,
    dim::Int = 1,
)::Nothing
    fam = QuadratureRuleSpec._common_rule_family(rule, dim)
    fam === :bspline || JobLoggerTools.error_benji(
        "ErrorBSplineRefinement only supports B-spline rules (got rule=$rule)"
    )
    return nothing
end

"""
    _require_bspline_inputs(
        N::Int,
        dim::Int,
        rule,
        boundary,
    ) -> Nothing

Validate the basic inputs required by the B-spline refinement estimator.

# Function description
This helper performs the common input checks used by the B-spline
refinement-based error-estimation layer. It verifies that the subdivision count
and dimensionality are valid, confirms that `rule` belongs to the B-spline,
and delegates boundary validation to `QuadratureBoundarySpec._decode_boundary`.

# Arguments
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Problem dimensionality.
- `rule`:
  B-spline quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Returns
- `nothing`

# Errors
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Throws if `rule` is not a supported B-spline rule.
- Throws if `boundary` is invalid.

# Notes
- This helper centralizes the shared validation logic for the public and
  internal B-spline refinement routines.
"""
@inline function _require_bspline_inputs(
    N::Int,
    dim::Int,
    rule,
    boundary,
)::Nothing
    (N >= 1) || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")
    _require_bspline_rule(rule, dim)
    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    return nothing
end

"""
    _quadrature_value_bspline(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule,
        boundary;
        λ = nothing,
        threaded_subgrid::Bool = false,
        real_type = nothing,
    ) -> Real

Evaluate the B-spline quadrature approximation of `f`.

# Function description
This helper validates the input configuration and then calls
`QuadratureDispatch.quadrature` to compute the quadrature approximation using
the requested B-spline rule, boundary condition, subdivision count, and
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
  Number of composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule`:
  B-spline quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter for smoothing B-spline rules. If `nothing`,
  zero is used in the active scalar type.
- `threaded_subgrid::Bool = false`:
  Whether to allow threaded subgrid evaluation in the quadrature backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and quadrature
  evaluation.

# Returns
- `Real`:
  The quadrature value produced by the selected evaluator, in the active scalar type.

# Errors
- Propagates validation errors from [`_require_bspline_inputs`](@ref).
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from `QuadratureDispatch.quadrature`.

# Notes
- This helper performs no derivative-based work.
- It is used internally by the refinement-based B-spline error estimator.
"""
@inline function _quadrature_value_bspline(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule,
    boundary;
    λ = nothing,
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

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    _require_bspline_inputs(N, dim, rule, boundary)

    q = QuadratureDispatch.quadrature(
        f,
        a isa AbstractVector || a isa Tuple ? map(x -> convert(T, x), a) : convert(T, a),
        b isa AbstractVector || b isa Tuple ? map(x -> convert(T, x), b) : convert(T, b),
        N,
        dim,
        rule,
        boundary;
        λ = λT,
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    return q
end

"""
    _estimate_by_refinement_bspline(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule,
        boundary;
        λ = nothing,
        threaded_subgrid::Bool = false,
        real_type = nothing,
        I_coarse = nothing,
    )

Estimate the B-spline quadrature error by comparing coarse and refined
composite quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
B-spline quadrature rules. It computes

- a coarse estimate at subdivision count `N`, either by evaluating it
  internally or by reusing the externally supplied `I_coarse`, and
- a refined estimate using `2N` composite blocks,

then forms their difference

```julia
diff = q_fine - q_coarse
```

Two domain conventions are supported:

* **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the mesh sizes are scalar quantities.

* **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, the mesh sizes are
  constructed componentwise and stored as per-axis tuples.

The returned named tuple stores both quadrature values, the corresponding mesh
sizes, and the absolute refinement difference used as the effective error
estimate.

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
* `rule`:
  B-spline quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
* `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments

* `λ = nothing`:
  Optional smoothing parameter for smoothing B-spline rules. If `nothing`,
  zero is used in the active scalar type.
* `threaded_subgrid::Bool = false`:
  Whether to allow threaded subgrid evaluation in the coarse and refined
  quadrature calls.
* `real_type = nothing`:
  Optional scalar type used internally for bound conversion, mesh sizes,
  and quadrature evaluation.
* `I_coarse = nothing`:
  Optional precomputed coarse quadrature value. If provided, the helper reuses
  it instead of recomputing the coarse B-spline quadrature value.

# Returns

* `NamedTuple` with fields:

  * `method`      : method tag `:bspline_refinement_difference`
  * `rule`        : quadrature rule specification
  * `boundary`    : boundary-condition specification
  * `N_coarse`    : coarse subdivision count
  * `N_fine`      : refined subdivision count (`2N`)
  * `dim`         : dimensionality
  * `h_coarse`    : coarse mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `h_fine`      : refined mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `q_coarse`    : coarse quadrature value, either reused from `I_coarse` or computed internally
  * `q_fine`      : refined quadrature value
  * `estimate`    : absolute refinement difference
  * `signed_diff` : signed refinement difference
  * `reference`   : refined quadrature value used as the internal reference

# Errors

* Throws if `rule` is not a supported B-spline rule.
* Throws if `N < 1`.
* Throws if `dim < 1`.
* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
* Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
* Propagates errors from the quadrature-evaluation layer.

# Notes

* This estimator does not use derivatives or residual moments.
* The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
* The optional `I_coarse` keyword is intended to avoid redundant coarse
  quadrature work when the caller has already computed that value.
* Axis-wise `rule` specifications must remain within the B-spline family on
  every axis.
"""
function _estimate_by_refinement_bspline(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule,
    boundary;
    λ = nothing,
    threaded_subgrid::Bool = false,
    real_type = nothing,
    I_coarse = nothing,
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

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    _require_bspline_inputs(N, dim, rule, boundary)

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

    q_coarse = isnothing(I_coarse) ?
        _quadrature_value_bspline(
            f,
            aa,
            bb,
            N,
            dim,
            rule,
            boundary;
            λ = λT,
            threaded_subgrid = threaded_subgrid,
            real_type = T,
        ) :
        I_coarse

    q_fine = _quadrature_value_bspline(
        f,
        aa,
        bb,
        2N,
        dim,
        rule,
        boundary;
        λ = λT,
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    diff = q_fine - q_coarse

    return (;
        method      = :bspline_refinement_difference,
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
    error_estimate_refinement_bspline(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ = nothing,
        threaded_subgrid::Bool = false,
        real_type = nothing,
        I_coarse = nothing,
    )

Unified public dispatcher for B-spline refinement-based error estimation.

# Function description
This function provides the main B-spline-specific entry point for the
refinement-based error-estimation layer.

It supports both of the following domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds.

The routine validates the rule family and boundary selector, then dispatches to
[`_estimate_by_refinement_bspline`](@ref).

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
  B-spline quadrature rule specification.
  This may be either a scalar rule symbol or a tuple/vector of per-axis rule
  symbols of length `dim`.
- `boundary`:
  Boundary-condition specification.
  This may be either a scalar boundary symbol or a tuple/vector of per-axis
  boundary symbols of length `dim`.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter for smoothing B-spline rules. If `nothing`,
  zero is used in the active scalar type.
- `threaded_subgrid::Bool = false`:
  Whether to allow threaded subgrid evaluation in the underlying refinement calls.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and refinement
  evaluation.
- `I_coarse = nothing`:
  Optional precomputed coarse quadrature value forwarded to
  [`_estimate_by_refinement_bspline`](@ref).

# Returns
- The named tuple produced by the selected refinement estimator.

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from the selected refinement routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on a derivative backend or jet construction.
- The optional `I_coarse` keyword exists to let callers reuse an already
  computed coarse quadrature value.
"""
function error_estimate_refinement_bspline(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ = nothing,
    threaded_subgrid::Bool = false,
    real_type = nothing,
    I_coarse = nothing,
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

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    _require_bspline_rule(rule, dim)

    return _estimate_by_refinement_bspline(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ = λT,
        threaded_subgrid = threaded_subgrid,
        real_type = T,
        I_coarse = I_coarse,
    )
end

end  # module ErrorBSplineRefinement
