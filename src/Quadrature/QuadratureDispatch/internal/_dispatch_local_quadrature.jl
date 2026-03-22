# ============================================================================
# src/Quadrature/QuadratureDispatch/internal/_dispatch_local_quadrature.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dispatch_local_quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ,
        real_type,
    ) -> Real

Dispatch to the local CPU quadrature implementation selected by `dim`.

# Function description
This helper contains the dimension-based selection logic for the non-CUDA,
non-threaded local quadrature path.

It forwards the call to one of:

- [`quadrature_1d`](@ref)
- [`quadrature_2d`](@ref)
- [`quadrature_3d`](@ref)
- [`quadrature_4d`](@ref)
- [`quadrature_nd`](@ref)

without performing separate backend selection of its own.

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
- `real_type`:
  Active scalar type.

# Returns
- `Real`:
  Estimated integral value returned by the selected local quadrature routine.

# Errors
- Propagates any error thrown by the selected local quadrature routine.

# Notes
- This helper is internal to the public dispatcher and does not perform CUDA or
  threaded backend selection.
"""
function _dispatch_local_quadrature(
    integrand,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ,
    real_type,
)
    if dim == 1
        return quadrature_1d(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 2
        return quadrature_2d(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 3
        return quadrature_3d(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            λ = λ,
            real_type = real_type,
        )
    elseif dim == 4
        return quadrature_4d(
            integrand,
            a,
            b,
            N,
            rule,
            boundary;
            λ = λ,
            real_type = real_type,
        )
    else
        return quadrature_nd(
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
    end
end
