# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_direct.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_direct(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        real_type = nothing,
    )

Unified interface for estimating an axis-separable midpoint-residual truncation-error model.

# Function description
This is the public non-threaded dispatcher for the direct derivative-based
error-estimation layer.

It routes to the matching dimension-specific estimator:

- [`error_estimate_derivative_direct_1d`](@ref) for `dim == 1`
- [`error_estimate_derivative_direct_2d`](@ref) for `dim == 2`
- [`error_estimate_derivative_direct_3d`](@ref) for `dim == 3`
- [`error_estimate_derivative_direct_4d`](@ref) for `dim == 4`
- [`error_estimate_derivative_direct_nd`](@ref) otherwise

All implementations share the same residual-term extraction logic and the same
derivative-backend interface via [`AutoDerivativeDirect.nth_derivative`](@ref).

Both hypercube-style scalar bounds and axis-wise rectangular bounds are
supported. Rectangular-domain support is provided by the selected
dimension-specific backend.

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
  Number of subintervals per axis.
- `dim`:
  Number of dimensions.
- `rule`:
  Quadrature rule specification.
  This may be either a scalar rule symbol shared across all axes, or a
  tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary pattern specification.
  This may be either a scalar boundary symbol shared across all axes, or a
  tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `err_method`:
  Derivative backend selector passed to [`AutoDerivativeDirect.nth_derivative`](@ref).
- `nerr_terms`:
  Number of nonzero residual terms to include.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and downstream
  derivative-estimator evaluation.

# Returns
- Same return object as the selected dimension-specific estimator.

# Errors
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from the selected estimator.
"""
function error_estimate_derivative_direct(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    real_type = nothing,
)
    T = _resolve_error_estimate_derivative_type(
        a,
        b,
        dim,
        real_type,
    )
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    return _dispatch_error_estimate_derivative_direct_by_dim(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method = err_method,
        nerr_terms = nerr_terms,
        real_type = T,
    )
end
