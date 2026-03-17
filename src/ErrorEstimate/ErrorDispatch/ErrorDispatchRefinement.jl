# ============================================================================
# src/ErrorEstimate/ErrorDispatchRefine.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorDispatchRefine

import ..JobLoggerTools
import ..Quadrature.Gauss
import ..Quadrature.NewtonCotes
import ..Quadrature.BSpline

import ..ErrorGaussRefine
import ..ErrorNewtonCotesRefine
import ..ErrorBSplineRefine


"""
    _dispatch_refine(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ::Float64 = 0.0,
    )

Dispatch a refinement-based error-estimation request to the matching quadrature
family backend.

# Function description
This internal helper provides the rule-family dispatch layer for the unified
refinement-based error-estimation interface.

It selects the backend according to `rule`:

- Gauss-family rules → [`ErrorGaussRefine.error_estimate_gauss`](@ref)
- Newton-Cotes rules → [`ErrorNewtonCotesRefine.error_estimate_newton_cotes`](@ref)
- B-spline rules → [`ErrorBSplineRefine.error_estimate_bspline`](@ref)

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
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter forwarded only to the B-spline refinement backend.

# Returns
- The named tuple returned by the selected rule-family refinement estimator.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `rule` is not supported by the
  refinement-dispatch layer.
- Propagates errors from the selected backend estimator.

# Notes
- This helper is internal and performs family-level routing only.
- The `λ` keyword is ignored for non-B-spline rules.
"""
@inline function _dispatch_refine(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ::Float64 = 0.0,
)
    if Gauss._is_gauss_rule(rule)
        return ErrorGaussRefine.error_estimate_gauss(
            f, a, b, N, dim, rule, boundary
        )

    elseif NewtonCotes._is_newton_cotes_rule(rule)
        return ErrorNewtonCotesRefine.error_estimate_newton_cotes(
            f, a, b, N, dim, rule, boundary
        )

    elseif BSpline._is_bspline_rule(rule)
        return ErrorBSplineRefine.error_estimate_bspline(
            f, a, b, N, dim, rule, boundary; λ=λ
        )

    else
        JobLoggerTools.error_benji(
            "Refinement error estimator does not support rule=$rule"
        )
    end
end

"""
    error_estimate_refine(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ::Float64 = 0.0,
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
  Quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter forwarded only when `rule` belongs to the B-spline family.

# Returns
- The named tuple produced by the selected refinement backend.

# Errors
- Throws if `rule` does not belong to a supported refinement backend.
- Propagates validation and computation errors from the selected backend.

# Notes
- This dispatcher is refinement-only and does not use derivative backends,
  derivative jets, or residual-moment models.
- For non-B-spline rules, the `λ` keyword is accepted for interface uniformity
  but is not used.
"""
function error_estimate_refine(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ::Float64 = 0.0,
)
    return _dispatch_refine(
        f, a, b, N, dim, rule, boundary; λ=λ
    )
end

end  # module ErrorDispatchRefine