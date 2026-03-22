# ============================================================================
# src/ErrorEstimate/ErrorDispatch/internal/_dispatch_error_estimate_backend.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_error_estimate_backend(
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
        λ,
        threaded_subgrid::Bool,
        real_type,
        I_coarse,
    )

Dispatch a unified error-estimation request to the selected backend family.

# Function description
This helper centralizes the top-level strategy selection for the public
[`error_estimate`](@ref) entry point.

The current rule is:

1. if `err_method === :refinement`, use
   [`ErrorDispatchRefinement.error_estimate_refinement`](@ref)
2. otherwise, use the derivative branch selected by
   [`_dispatch_derivative_error_estimate`](@ref)

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
  Error-estimation backend selector.
- `nerr_terms::Int`:
  Number of residual terms for derivative-based estimation.
- `use_error_jet::Bool`:
  Whether to use the derivative-jet backend.
- `λ`:
  Normalized refinement parameter.
- `threaded_subgrid::Bool`:
  Whether refinement backends may use CPU threaded subgrid execution.
- `real_type`:
  Active scalar type.
- `I_coarse`:
  Optional coarse quadrature value forwarded only to refinement backends.

# Returns
- The named tuple returned by the selected backend.

# Errors
- Propagates validation and computation errors from the selected backend.
"""
function _dispatch_error_estimate_backend(
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
    λ,
    threaded_subgrid::Bool,
    real_type,
    I_coarse,
)
    if err_method === :refinement
        return ErrorDispatchRefinement.error_estimate_refinement(
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

    return _dispatch_derivative_error_estimate(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method = err_method,
        nerr_terms = nerr_terms,
        use_error_jet = use_error_jet,
        real_type = real_type,
    )
end
