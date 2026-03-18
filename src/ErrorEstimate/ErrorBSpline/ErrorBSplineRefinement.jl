# ============================================================================
# src/ErrorEstimate/ErrorBSplineRefinement.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorBSplineRefinement

import ..JobLoggerTools
import ..BSpline
import ..QuadratureUtils
import ..QuadratureDispatch

"""
    _require_bspline_rule(
        rule::Symbol
    ) -> Nothing

Validate that `rule` belongs to the supported B-spline quadrature family.

# Function description
This internal helper checks whether the supplied quadrature-rule symbol is a
B-spline rule recognized by the quadrature layer. It is used as a guard before
calling B-spline-specific parsing or node/weight construction routines.

# Arguments
- `rule::Symbol`:
  Quadrature rule symbol to validate.

# Returns
- `nothing`

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a supported
  B-spline rule.

# Notes
- This helper performs only rule-family validation.
- It does not validate `boundary`, `N`, `dim`, or the smoothing parameter `λ`.
"""
@inline function _require_bspline_rule(
    rule::Symbol
)::Nothing
    BSpline._is_bspline_rule(rule) ||
        JobLoggerTools.error_benji("ErrorBSplineDerivative only supports B-spline rules (got rule=$rule)")
    return nothing
end

"""
    _require_bspline_inputs(
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol,
    ) -> Nothing

Validate the basic inputs required by the B-spline refinement estimator.

# Function description
This helper performs the common input checks used by the B-spline
refinement-based error-estimation layer. It verifies that the subdivision count
and dimensionality are valid, confirms that `rule` belongs to the B-spline,
and delegates boundary validation to `QuadratureUtils._decode_boundary`.

# Arguments
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Problem dimensionality.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- `nothing`

# Errors
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Throws if `rule` is not a supported B-spline rule.
- Throws if `boundary` is invalid.

# Notes
- This helper centralizes the shared validation logic for the public and
  internal Gauss refinement routines.
"""
@inline function _require_bspline_inputs(
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol,
)::Nothing
    (N >= 1) || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")
    _require_bspline_rule(rule)
    QuadratureUtils._decode_boundary(boundary)
    return nothing
end

"""
    _quadrature_value_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Dispatch to the appropriate dimension-specific B-spline quadrature evaluator.

# Function description
This helper selects the specialized B-spline quadrature evaluator matching
`dim`.

# Arguments
- `f`:
  Scalar integrand callable.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The quadrature value produced by the selected evaluator.

# Errors
- Propagates errors from the selected dimension-specific routine.

# Notes
- This function only dispatches; it does not implement a separate quadrature
  algorithm.
"""
@inline function _quadrature_value_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
    threaded_subgrid::Bool = false
)::Float64
    _require_bspline_inputs(N, dim, rule, boundary)

    q = QuadratureDispatch.quadrature(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ=λ,
        threaded_subgrid = threaded_subgrid
    )

    return float(q)
end

"""
    _estimate_by_refinement_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the B-spline quadrature error by comparing coarse and refined
composite quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
B-spline quadrature rules. It computes

- a coarse estimate using `N` composite blocks, and
- a refined estimate using `2N` composite blocks,

then forms their difference

```julia
diff = q_fine - q_coarse.
```

The returned named tuple stores both quadrature values, the corresponding mesh
sizes, and the absolute refinement difference used as the effective error
estimate.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Coarse subdivision count.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `NamedTuple` with fields:
  - `method`      : method tag `:bspline_refinement_difference`
  - `rule`        : quadrature rule symbol
  - `boundary`    : boundary-condition symbol
  - `N_coarse`    : coarse subdivision count
  - `N_fine`      : refined subdivision count (`2N`)
  - `dim`         : dimensionality
  - `h_coarse`    : coarse mesh size
  - `h_fine`      : refined mesh size
  - `q_coarse`    : coarse quadrature value
  - `q_fine`      : refined quadrature value
  - `estimate`    : absolute refinement difference
  - `signed_diff` : signed refinement difference
  - `reference`   : refined quadrature value used as the internal reference

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Propagates errors from the quadrature-evaluation layer.

# Notes
- This estimator does not use derivatives or residual moments.
- The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
"""
function _estimate_by_refinement_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
    threaded_subgrid::Bool = false
)
    _require_bspline_inputs(rule, dim, rule,boundary)

    aa = float(a)
    bb = float(b)

    h_coarse = (bb - aa) / N
    h_fine   = (bb - aa) / (2N)

    q_coarse = _quadrature_value_bspline(
        f, 
        aa, 
        bb, 
        N,  
        dim, 
        rule, 
        boundary; 
        λ=λ,
        threaded_subgrid=threaded_subgrid
    )
    q_fine   = _quadrature_value_bspline(
        f, 
        aa, 
        bb, 
        2N, 
        dim, 
        rule, 
        boundary; 
        λ=λ,
        threaded_subgrid=threaded_subgrid
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
        λ::Float64 = 0.0
    )

Unified public dispatcher for B-spline refinement-based error estimation.

# Function description
This function provides the main B-spline-specific entry point for the
refinement-based error-estimation layer. It dispatches to the dimension-specific
specializations:

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound.
- `b`:
  Upper integration bound.
- `N`:
  Coarse subdivision count.
- `dim`:
  Number of dimensions.
- `rule`:
  B-spline quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- The named tuple produced by the selected dimension-specific refinement
  estimator.

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Propagates errors from the selected dimension-specific routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on a derivative backend or jet construction.
"""
function error_estimate_refinement_bspline(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ::Float64 = 0.0,
    threaded_subgrid::Bool = false
)
    _require_bspline_rule(rule)
    QuadratureUtils._decode_boundary(boundary)

    return _estimate_by_refinement_bspline(
        f, 
        a, 
        b, 
        N, 
        dim, 
        rule, 
        boundary; 
        λ=λ,
        threaded_subgrid=threaded_subgrid
    )
end

end  # module ErrorBSplineRefinement