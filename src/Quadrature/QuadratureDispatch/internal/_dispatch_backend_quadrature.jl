# ============================================================================
# src/Quadrature/QuadratureDispatch/internal/_dispatch_backend_quadrature.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_backend_quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ,
        use_cuda::Bool,
        threaded_subgrid::Bool,
        real_type,
    ) -> Real

Dispatch the public quadrature request to the selected execution backend.

# Function description
This helper centralizes backend-strategy selection for the public
[`quadrature`](@ref) entry point.

The current priority order is:

1. CUDA backend when `use_cuda == true`
2. threaded-subgrid CPU backend when `threaded_subgrid == true`
3. local CPU dimension-dispatched backend otherwise

# Arguments
- `integrand`:
  Callable integrand.
- `a`, `b`:
  Integration-bound specifications.
- `N`:
  Subdivision or block count.
- `dim`:
  Problem dimensionality.
- `rule`:
  Quadrature-rule specification.
- `boundary`:
  Boundary specification.

# Keyword arguments
- `λ`:
  Normalized optional rule parameter.
- `use_cuda::Bool`:
  Whether to prefer the CUDA backend.
- `threaded_subgrid::Bool`:
  Whether to prefer the threaded-subgrid CPU backend when CUDA is disabled.
- `real_type`:
  Active scalar type.

# Returns
- `Real`:
  Estimated integral value returned by the selected backend.

# Errors
- Propagates any error thrown by the selected backend implementation.

# Notes
- This helper performs strategy selection only.
- Local CPU dispatch is delegated to [`_dispatch_local_quadrature`](@ref).
"""
function _dispatch_backend_quadrature(
    integrand,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ,
    use_cuda::Bool,
    threaded_subgrid::Bool,
    real_type,
)
    if use_cuda
        return QuadratureDispatchCUDA.quadrature_cuda(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            λ = λ,
            real_type = real_type,
        )
    elseif threaded_subgrid
        return QuadratureDispatchThreadedSubgrid.quadrature_threaded_subgrid(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            λ = λ,
            real_type = real_type,
        )
    else
        return _dispatch_local_quadrature(
            integrand,
            a,
            b,
            N,
            dim,
            rule,
            boundary;
            λ = λ,
            real_type = real_type,
        )
    end
end
