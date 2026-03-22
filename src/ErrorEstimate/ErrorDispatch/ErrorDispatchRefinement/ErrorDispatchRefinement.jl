# ============================================================================
# src/ErrorEstimate/ErrorDispatchRefinement.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorDispatchRefinement

Unified refinement-based error-estimation dispatch layer.

# Module description
`ErrorDispatchRefinement` provides the shared rule-family dispatch used by the
refinement-based error estimator.

It validates the incoming quadrature-rule specification, enforces the current
same-family restriction for axis-wise refinement rules, and forwards the
request to the matching Gauss, Newton-Cotes, or B-spline backend.

# Notes
- This is an internal module.
- Public callers should use the higher-level `ErrorDispatch.error_estimate`
  interface rather than invoking these helpers directly.
"""
module ErrorDispatchRefinement

import ..JobLoggerTools
import ..QuadratureRuleSpec
import ..NewtonCotes
import ..Gauss
import ..BSpline
import ..ErrorGaussRefinement
import ..ErrorNewtonCotesRefinement
import ..ErrorBSplineRefinement

"""
    _dispatch_refinement(
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
Dispatch a refinement-based error-estimation request to the matching quadrature
family backend.

# Function description
This internal helper provides the rule-family dispatch layer for the unified
refinement-based error-estimation interface.

It selects the backend according to `rule`:

- Gauss-family rules → [`ErrorGaussRefinement.error_estimate_refinement_gauss`](@ref)
- Newton-Cotes rules → [`ErrorNewtonCotesRefinement.error_estimate_refinement_newton_cotes`](@ref)
- B-spline rules → [`ErrorBSplineRefinement.error_estimate_refinement_bspline`](@ref)

If the rule does not belong to any supported refinement backend, an error is
raised.

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
  Quadrature rule specification.
  This may be either a scalar rule symbol shared across all axes, or a
  tuple/vector of per-axis rule symbols of length `dim`. Axis-wise rule
  specifications must belong to a single quadrature family.
- `boundary`:
  Boundary-condition specification.
  This may be either a scalar boundary symbol shared across all axes, or a
  tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter forwarded only to the B-spline refinement backend.
  If `nothing`, zero is used in the active scalar type.
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the selected refinement backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and backend dispatch.
- `I_coarse = nothing`:
  Optional precomputed coarse quadrature value forwarded to the selected
  refinement backend so the coarse-grid quadrature value can be reused.

# Returns
- The named tuple returned by the selected rule-family refinement estimator.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `rule` is not supported by the
  refinement-dispatch layer.
- Throws `ArgumentError` if an axis-wise `rule` specification mixes multiple
  quadrature families.
- Propagates errors from the selected backend estimator.

# Notes
- This helper is internal and performs family-level routing only.
- The `λ` keyword is ignored for non-B-spline rules.
- The `I_coarse` keyword is forwarded unchanged to whichever refinement backend
  is selected.
"""
@inline function _dispatch_refinement(
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
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    family = QuadratureRuleSpec._common_rule_family(rule, dim)

    if family === :gauss
        return ErrorGaussRefinement.error_estimate_refinement_gauss(
            f, a, b, N, dim, rule, boundary;
            threaded_subgrid = threaded_subgrid,
            real_type = T,
            I_coarse = I_coarse,
        )

    elseif family === :newton_cotes
        return ErrorNewtonCotesRefinement.error_estimate_refinement_newton_cotes(
            f, a, b, N, dim, rule, boundary;
            threaded_subgrid = threaded_subgrid,
            real_type = T,
            I_coarse = I_coarse,
        )

    elseif family === :bspline
        return ErrorBSplineRefinement.error_estimate_refinement_bspline(
            f, a, b, N, dim, rule, boundary;
            λ = λT,
            threaded_subgrid = threaded_subgrid,
            real_type = T,
            I_coarse = I_coarse,
        )

    else
        JobLoggerTools.error_benji(
            "Refinement error estimator does not support rule=$rule"
        )
    end
end

"""
    error_estimate_refinement(
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

Unified public dispatcher for refinement-based error estimation across all
supported quadrature families.

# Function description
This function provides the main public entry point for the refinement-based
error-estimation layer. It delegates the request to the appropriate backend
according to the rule family:

- Gauss-family rules
- Newton-Cotes rules
- B-spline rules

The selected backend then performs a coarse-versus-refined quadrature
comparison and returns its rule-specific refinement estimate object.

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
  Quadrature rule specification.
  This may be either a scalar rule symbol shared across all axes, or a
  tuple/vector of per-axis rule symbols of length `dim`. Axis-wise rule
  specifications must belong to a single quadrature family.
- `boundary`:
  Boundary-condition specification.
  This may be either a scalar boundary symbol shared across all axes, or a
  tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter forwarded only when `rule` belongs to the B-spline family.
  If `nothing`, zero is used in the active scalar type.
- `threaded_subgrid::Bool = false`:
  Whether to allow CPU threaded subgrid execution in the selected refinement backend.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and refinement dispatch.
- `I_coarse = nothing`:
  Optional precomputed coarse quadrature value forwarded to the selected
  backend so the coarse-grid quadrature value can be reused.

# Returns
- The named tuple produced by the selected refinement backend.

# Errors
- Throws if `rule` does not belong to a supported refinement backend.
- Throws if an axis-wise `rule` specification mixes multiple quadrature
  families.
- Propagates validation and computation errors from the selected backend.

# Notes
- This dispatcher is refinement-only and does not use derivative backends,
  derivative jets, or residual-moment models.
- For non-B-spline rules, the `λ` keyword is accepted for interface uniformity
  but is not used.
- The active scalar type can be overridden through `real_type`.
- The `I_coarse` keyword exists to avoid redundant coarse quadrature work when
  the caller already has that value.
"""
function error_estimate_refinement(
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
    return _dispatch_refinement(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ = λ,
        threaded_subgrid = threaded_subgrid,
        real_type = real_type,
        I_coarse = I_coarse,
    )
end

end  # module ErrorDispatchRefinement
