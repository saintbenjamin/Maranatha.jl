# ============================================================================
# src/ErrorEstimate/ErrorNewtonCotesRefinement.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorNewtonCotesRefinement

Refinement-based error-estimation backend for Newton-Cotes quadrature rules.

# Module description
`ErrorNewtonCotesRefinement` implements the Newton-Cotes branch of the
refinement-based error estimator.

It validates Newton-Cotes rule specifications, computes a refinement level that
is admissible across all active axes, evaluates coarse and refined quadrature
values, and packages the resulting quadrature-difference estimate in a uniform
result structure understood by the higher-level error-dispatch layer.

# Notes
- This is an internal module.
- Axis-wise `rule` specifications are supported here when all axes remain in
  the Newton-Cotes family.
"""
module ErrorNewtonCotesRefinement

import ..JobLoggerTools
import ..QuadratureBoundarySpec
import ..QuadratureRuleSpec
import ..QuadratureDispatch
import ..NewtonCotes

"""
    _require_newton_cotes_rule(
        rule,
        dim::Int = 1,
    ) -> Nothing

Validate that `rule` belongs to the supported Newton-Cotes quadrature family.

# Function description
This internal helper checks whether the supplied quadrature-rule specification
belongs to the Newton-Cotes family recognized by the quadrature layer. It is
used as a guard before calling Newton-Cotes-specific refinement routines.

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
  Newton-Cotes rule.

# Notes
- This helper validates only the rule family.
- It does not validate `boundary`, `N`, or `dim`.
- Axis-wise rule specifications must use Newton-Cotes rules on every axis.
"""
@inline function _require_newton_cotes_rule(
    rule,
    dim::Int = 1,
)::Nothing
    fam = QuadratureRuleSpec._common_rule_family(rule, dim)
    fam === :newton_cotes || JobLoggerTools.error_benji(
        "ErrorNewtonCotesRefinement only supports Newton-Cotes rules (got rule=$rule)"
    )
    return nothing
end

"""
    _require_newton_cotes_inputs(
        N::Int,
        dim::Int,
        rule,
        boundary,
    ) -> Nothing

Validate the basic inputs required by the Newton-Cotes refinement estimator.

# Function description
This helper performs the common input checks used by the Newton-Cotes
refinement-based error-estimation layer. It verifies that the subdivision count
and dimensionality are valid, confirms that `rule` belongs to the Newton-Cotes
family, and delegates boundary validation to `QuadratureBoundarySpec._decode_boundary`.

# Arguments
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Problem dimensionality.
- `rule`:
  Newton-Cotes quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Returns
- `nothing`

# Errors
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Throws if `rule` is not a supported Newton-Cotes rule.
- Throws if `boundary` is invalid.

# Notes
- This helper centralizes the shared validation logic for the public and
  internal Newton-Cotes refinement routines.
"""
@inline function _require_newton_cotes_inputs(
    N::Int,
    dim::Int,
    rule,
    boundary,
)::Nothing
    (N >= 1)   || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")
    _require_newton_cotes_rule(rule, dim)
    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    return nothing
end

"""
    _next_valid_Nsub_common(
        rule,
        boundary,
        dim::Int,
        Ntarget::Int,
    ) -> Int

Return the smallest common Newton-Cotes subdivision count greater than or equal
to `Ntarget` that is valid on every axis.

# Function description
For axis-wise Newton-Cotes configurations, each axis may impose its own
admissibility constraint on the composite subdivision count. This helper
repeatedly applies [`NewtonCotes._next_valid_Nsub`](@ref) on every axis until a
common valid count is reached.

# Arguments
- `rule`:
  Newton-Cotes quadrature rule specification, scalar or axis-wise.
- `boundary`:
  Boundary specification, scalar or axis-wise.
- `dim::Int`:
  Problem dimensionality.
- `Ntarget::Int`:
  Requested lower bound for the refined subdivision count.

# Returns
- `Int`:
  Smallest common valid subdivision count `>= Ntarget`.

# Errors
- Propagates Newton-Cotes-family validation errors from
  [`_require_newton_cotes_rule`](@ref).
- Propagates boundary-access and admissibility errors from
  [`QuadratureRuleSpec._rule_at`](@ref),
  [`QuadratureBoundarySpec._boundary_at`](@ref), and
  [`NewtonCotes._next_valid_Nsub`](@ref).

# Notes
- This helper is used by the refinement estimator so that all axes share one
  common refined composite tiling.
"""
function _next_valid_Nsub_common(
    rule,
    boundary,
    dim::Int,
    Ntarget::Int,
)::Int
    _require_newton_cotes_rule(rule, dim)

    Ncand = Ntarget

    while true
        updated = false

        for d in 1:dim
            rd = QuadratureRuleSpec._rule_at(rule, d, dim)
            bd = QuadratureBoundarySpec._boundary_at(boundary, d, dim)
            p = NewtonCotes._parse_newton_p(rd)
            Nd = NewtonCotes._next_valid_Nsub(p, bd, Ncand)

            if Nd > Ncand
                Ncand = Nd
                updated = true
            end
        end

        updated || return Ncand
    end
end

"""
    _quadrature_value_newton_cotes(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule,
        boundary;
        threaded_subgrid::Bool = false,
        real_type = nothing,
    ) -> Real

Evaluate the Newton-Cotes quadrature approximation of `f`.

# Function description
This helper validates the input configuration and then calls
`QuadratureDispatch.quadrature` to compute the quadrature approximation using
the requested Newton-Cotes rule, boundary condition, subdivision count, and
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
- `rule`:
  Newton-Cotes quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the quadrature backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and quadrature
  evaluation.

# Returns
- `Real`:
  The quadrature value produced by the Newton-Cotes backend, in the active scalar type.

# Errors
- Propagates validation errors from [`_require_newton_cotes_inputs`](@ref).
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from `QuadratureDispatch.quadrature`.

# Notes
- This helper performs no derivative-based work.
- It is used internally by the refinement-based Newton-Cotes error estimator.
"""
@inline function _quadrature_value_newton_cotes(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule,
    boundary;
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

    _require_newton_cotes_inputs(N, dim, rule, boundary)

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
    _estimate_by_refinement_newton_cotes(
        f,
        a,
        b,
        N::Int,
        dim::Int,
        rule,
        boundary;
        threaded_subgrid::Bool = false,
        real_type = nothing,
        I_coarse = nothing,
    )

Estimate the Newton-Cotes quadrature error by comparing coarse and refined
quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
Newton-Cotes quadrature rules. It computes

- a coarse quadrature value at subdivision count `N`, either by evaluating it
  internally or by reusing the externally supplied `I_coarse`, and
- a refined quadrature value using a common axis-compatible refined
  subdivision count obtained from [`_next_valid_Nsub_common`](@ref),

then forms the refinement difference

```julia
diff = q_fine - q_coarse
```

Because some Newton-Cotes boundary patterns require specific valid composite
subdivision counts, the refined run may use an adjusted `N_fine` rather than
exactly `2N` for the actual quadrature evaluation.

Two domain conventions are supported:

* **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the mesh sizes are scalar quantities.

* **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, the mesh sizes are
  constructed componentwise and stored as per-axis tuples.

The returned named tuple records both mesh sizes and quadrature values, and uses
`abs(diff)` as the effective error estimate.

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
  Newton-Cotes quadrature rule specification. This may be either a scalar rule
  symbol or a tuple/vector of per-axis rule symbols of length `dim`.
* `boundary`:
  Boundary-condition specification. This may be either a scalar boundary
  symbol or a tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments

* `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the coarse and refined
  quadrature calls.
* `real_type = nothing`:
  Optional scalar type used internally for bound conversion, mesh sizes,
  and quadrature evaluation.
* `I_coarse = nothing`:
  Optional precomputed coarse quadrature value. If provided, the helper reuses
  it instead of recomputing the coarse Newton-Cotes quadrature value.

# Returns

* `NamedTuple` with fields:

  * `method`      : method tag `:newton_cotes_refinement_difference`
  * `rule`        : quadrature rule specification
  * `boundary`    : boundary-condition specification
  * `N_coarse`    : coarse subdivision count
  * `N_fine`      : actual valid refined subdivision count used in the refined run
  * `dim`         : dimensionality
  * `h_coarse`    : coarse mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `h_fine`      : refined mesh size (scalar for hypercubes, per-axis tuple for rectangular domains)
  * `q_coarse`    : coarse quadrature value, either reused from `I_coarse` or computed internally
  * `q_fine`      : refined quadrature value
  * `estimate`    : absolute refinement difference
  * `signed_diff` : signed refinement difference
  * `reference`   : refined quadrature value used as the internal reference

# Errors

* Propagates validation errors from [`_require_newton_cotes_inputs`](@ref).
* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
* Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
* Propagates errors from the quadrature-evaluation layer.
* Propagates errors from the refined-subdivision adjustment logic.

# Notes

* This estimator does not use derivatives, jets, or residual moments.
* The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
* The `N_fine` field stores the actual valid refined subdivision count used in
  the refined quadrature evaluation.
* The optional `I_coarse` keyword is intended to avoid redundant coarse
  quadrature work when the caller has already computed that value.
* For axis-wise Newton-Cotes inputs, `N_fine` is chosen so that all axes share
  one common valid refined composite tiling.
"""
function _estimate_by_refinement_newton_cotes(
    f,
    a,
    b,
    N::Int,
    dim::Int,
    rule,
    boundary;
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

    _require_newton_cotes_inputs(N, dim, rule, boundary)

    N_fine_actual = _next_valid_Nsub_common(rule, boundary, dim, 2N)

    if a isa AbstractVector || a isa Tuple
        aa = ntuple(i -> convert(T, a[i]), dim)
        bb = ntuple(i -> convert(T, b[i]), dim)

        h_coarse = ntuple(i -> (bb[i] - aa[i]) / T(N), dim)
        h_fine   = ntuple(i -> (bb[i] - aa[i]) / T(N_fine_actual), dim)
    else
        aa = convert(T, a)
        bb = convert(T, b)

        h_coarse = (bb - aa) / T(N)
        h_fine   = (bb - aa) / T(N_fine_actual)
    end

    q_coarse = isnothing(I_coarse) ?
        _quadrature_value_newton_cotes(
            f,
            aa,
            bb,
            N,
            dim,
            rule,
            boundary;
            threaded_subgrid = threaded_subgrid,
            real_type = T,
        ) :
        I_coarse

    q_fine = _quadrature_value_newton_cotes(
        f,
        aa,
        bb,
        N_fine_actual,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    diff = q_fine - q_coarse

    return (;
        method      = :newton_cotes_refinement_difference,
        rule        = rule,
        boundary    = boundary,
        N_coarse    = N,
        N_fine      = N_fine_actual,
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
    error_estimate_refinement_newton_cotes(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid::Bool = false,
        real_type = nothing,
        I_coarse = nothing,
    )

Unified public dispatcher for Newton-Cotes refinement-based error estimation.

# Function description
This function provides the main Newton-Cotes-specific entry point for the
refinement-based error-estimation layer.

It supports both of the following domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds.

The routine validates the rule family and boundary selector, then dispatches to
[`_estimate_by_refinement_newton_cotes`](@ref).

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
  Newton-Cotes quadrature rule specification.
  This may be either a scalar rule symbol or a tuple/vector of per-axis rule
  symbols of length `dim`.
- `boundary`:
  Boundary-condition specification.
  This may be either a scalar boundary symbol or a tuple/vector of per-axis
  boundary symbols of length `dim`.

# Keyword arguments
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the quadrature backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and quadrature/error
  evaluation.
- `I_coarse = nothing`:
  Optional precomputed coarse quadrature value forwarded to
  [`_estimate_by_refinement_newton_cotes`](@ref).

# Returns
- The named tuple produced by the selected refinement estimator.

# Errors
- Throws if `rule` is not a supported Newton-Cotes rule.
- Throws if `boundary` is invalid.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from the refinement-estimation routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on derivative backends or jet construction.
- The optional `I_coarse` keyword exists to let callers reuse an already
  computed coarse quadrature value.
"""
function error_estimate_refinement_newton_cotes(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
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

    _require_newton_cotes_rule(rule, dim)

    return _estimate_by_refinement_newton_cotes(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        threaded_subgrid = threaded_subgrid,
        real_type = T,
        I_coarse = I_coarse,
    )
end

end  # module ErrorNewtonCotesRefinement
