# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_dispatch_error_estimate_derivative_direct_by_dim.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_error_estimate_derivative_direct_by_dim(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol,
        nerr_terms::Int,
        real_type,
    )

Dispatch the direct derivative-based error estimator to the dimension-specific
implementation selected by `dim`.

# Function description
This helper centralizes dimension-based selection for
[`error_estimate_derivative_direct`](@ref).

# Arguments
- `f`:
  Integrand callable.
- `a`, `b`:
  Integration-bound specifications.
- `N`:
  Quadrature subdivision count.
- `dim`:
  Problem dimensionality.
- `rule`:
  Quadrature-rule specification.
- `boundary`:
  Boundary specification.

# Keyword arguments
- `err_method::Symbol`:
  Derivative backend selector.
- `nerr_terms::Int`:
  Number of residual terms requested.
- `real_type`:
  Active scalar type.

# Returns
- The named tuple returned by the selected dimension-specific direct estimator.

# Errors
- Propagates any validation or computation error thrown by the selected
  dimension-specific direct estimator.
"""
function _dispatch_error_estimate_derivative_direct_by_dim(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol,
    nerr_terms::Int,
    real_type,
)
    if dim == 1
        return error_estimate_derivative_direct_1d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    elseif dim == 2
        return error_estimate_derivative_direct_2d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    elseif dim == 3
        return error_estimate_derivative_direct_3d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    elseif dim == 4
        return error_estimate_derivative_direct_4d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    else
        return error_estimate_derivative_direct_nd(
            f, a, b, N, rule, boundary;
            dim = dim,
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    end
end
