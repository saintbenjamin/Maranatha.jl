# ============================================================================
# src/ErrorEstimate/ErrorDispatch/internal/_dispatch_derivative_error_estimate.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_derivative_error_estimate(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol,
        nerr_terms::Int,
        use_error_jet::Bool,
        real_type,
    )

Dispatch a derivative-based error-estimation request to the direct or jet
backend.

# Function description
This helper centralizes the branch selection between

- [`ErrorDispatchDerivative.error_estimate_derivative_direct`](@ref), and
- [`ErrorDispatchDerivative.error_estimate_derivative_jet`](@ref)

using the `use_error_jet` flag.

# Arguments
- `f`:
  Integrand callable.
- `a`, `b`:
  Integration bounds.
- `N`:
  Subdivision count.
- `dim`:
  Number of dimensions.
- `rule`:
  Quadrature rule specification.
- `boundary`:
  Boundary specification.

# Keyword arguments
- `err_method::Symbol`:
  Derivative backend selector.
- `nerr_terms::Int`:
  Number of residual terms requested.
- `use_error_jet::Bool`:
  Whether to use the jet-based derivative path.
- `real_type`:
  Active scalar type.

# Returns
- The named tuple returned by the selected derivative backend.

# Errors
- Propagates validation and computation errors from the selected derivative
  backend.
"""
function _dispatch_derivative_error_estimate(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol,
    nerr_terms::Int,
    use_error_jet::Bool,
    real_type,
)
    if use_error_jet
        return ErrorDispatchDerivative.error_estimate_derivative_jet(
            f,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    else
        return ErrorDispatchDerivative.error_estimate_derivative_direct(
            f,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = real_type,
        )
    end
end
