# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_dispatch_threaded_subgrid_by_dim.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_threaded_subgrid_by_dim(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        nthreads_req::Int,
        λ,
        real_type,
    ) -> Real

Dispatch the threaded-subgrid backend to the dimension-specific implementation
selected by `dim`.

# Function description
This helper contains the dimension-based selection logic for
[`quadrature_threaded_subgrid`](@ref).

It forwards the call to one of:

- [`quadrature_1d_threaded_subgrid`](@ref)
- [`quadrature_2d_threaded_subgrid`](@ref)
- [`quadrature_3d_threaded_subgrid`](@ref)
- [`quadrature_4d_threaded_subgrid`](@ref)
- [`quadrature_nd_threaded_subgrid`](@ref)

without performing separate scalar-type preparation of its own.

# Arguments
- `f`:
  Integrand callable.
- `a`, `b`:
  Integration-bound specifications.
- `N`:
  Subdivision or block count.
- `rule`:
  Quadrature-rule specification.
- `boundary`:
  Boundary specification.

# Keyword arguments
- `dim::Int`:
  Problem dimensionality.
- `nthreads_req::Int`:
  Requested number of threads.
- `λ`:
  Normalized optional rule parameter.
- `real_type`:
  Active scalar type.

# Returns
- `Real`:
  Estimated integral value returned by the selected threaded-subgrid backend.

# Errors
- Propagates any error thrown by the selected dimension-specific routine.
"""
function _dispatch_threaded_subgrid_by_dim(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    nthreads_req::Int,
    λ,
    real_type,
)
    if dim == 1
        return quadrature_1d_threaded_subgrid(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            nthreads_req = nthreads_req,
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 2
        return quadrature_2d_threaded_subgrid(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            nthreads_req = nthreads_req,
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 3
        return quadrature_3d_threaded_subgrid(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            nthreads_req = nthreads_req,
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 4
        return quadrature_4d_threaded_subgrid(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            nthreads_req = nthreads_req,
            λ = λ,
            real_type = real_type,
        )
    else
        return quadrature_nd_threaded_subgrid(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            dim = dim,
            nthreads_req = nthreads_req,
            λ = λ,
            real_type = real_type,
        )
    end
end
