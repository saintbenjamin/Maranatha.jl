# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet(
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

Dispatch to the jet-based error estimator for the requested dimensionality.

# Function description
This function serves as a dimension-based dispatcher for the jet-oriented
error-estimation pipeline. It selects the appropriate backend according to
`dim`:

- `dim == 1` → [`error_estimate_derivative_jet_1d`](@ref)
- `dim == 2` → [`error_estimate_derivative_jet_2d`](@ref)
- `dim == 3` → [`error_estimate_derivative_jet_3d`](@ref)
- `dim == 4` → [`error_estimate_derivative_jet_4d`](@ref)
- otherwise  → [`error_estimate_derivative_jet_nd`](@ref) with `dim = dim`

Each dispatched routine uses derivative jets internally rather than requesting
scalar derivatives one by one.

Both hypercube-style scalar bounds and axis-wise rectangular bounds are
supported. Rectangular-domain support is provided by the selected
dimension-specific backend.

# Arguments
- `f`:
  Integrand or scalar callable to be analyzed.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds.
- `N`:
  Number of subdivisions.
- `dim`:
  Problem dimensionality.
- `rule`:
  Quadrature rule specification.
  This may be either a scalar rule symbol shared across all axes, or a
  tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`:
  Boundary-condition specification.
  This may be either a scalar boundary symbol shared across all axes, or a
  tuple/vector of per-axis boundary symbols of length `dim`.

# Keyword arguments
- `err_method::Symbol`:
  Derivative backend selector
  (`:forwarddiff | :taylorseries |  | :enzyme`).
- `nerr_terms::Int`:
  Number of residual contributions to retain when constructing the effective
  error estimate.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and downstream
  jet-estimator evaluation.

# Returns
- The return value produced by the selected dimension-specific jet estimator.

# Errors
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Propagates errors from the selected dimension-specific jet estimator.

# Notes
- This dispatcher does not implement the estimator logic itself; it only routes
  the request to the dimension-appropriate backend.
- For dimensions other than `1`, `2`, `3`, and `4`, the generic
  [`error_estimate_derivative_jet_nd`](@ref) path is used.
- This interface parallels the non-jet error-estimation dispatcher, but is
  specialized for jet-based derivative reuse.
"""
function error_estimate_derivative_jet(
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

    return _dispatch_error_estimate_derivative_jet_by_dim(
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
