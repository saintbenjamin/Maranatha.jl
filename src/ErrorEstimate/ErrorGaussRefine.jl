# ============================================================================
# src/ErrorEstimate/ErrorGaussRefine.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorGaussRefine

import ..JobLoggerTools
import ..Quadrature.Gauss
import ..Quadrature.QuadratureDispatch

"""
    _require_gauss_rule(
        rule::Symbol
    ) -> Nothing

Validate that `rule` belongs to the supported Gauss-family quadrature rules.

# Function description
This internal helper checks whether the supplied quadrature-rule symbol is a
Gauss-family rule recognized by the quadrature layer. It is used as a guard
before calling Gauss-specific refinement routines.

# Arguments
- `rule::Symbol`:
  Quadrature rule symbol to validate.

# Returns
- `nothing`

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a supported
  Gauss-family rule.

# Notes
- This helper validates only the rule family.
- It does not validate `boundary`, `N`, or `dim`.
"""
@inline function _require_gauss_rule(
    rule::Symbol
)::Nothing
    Gauss._is_gauss_rule(rule) ||
        JobLoggerTools.error_benji(
            "ErrorGaussRefine only supports Gauss-family rules (got rule=$rule)"
        )
    return nothing
end

"""
    _require_gauss_inputs(
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol,
    ) -> Nothing

Validate the basic inputs required by the Gauss refinement estimator.

# Function description
This helper performs the common input checks used by the Gauss-family
refinement-based error-estimation layer. It verifies that the subdivision count
and dimensionality are valid, confirms that `rule` belongs to the Gauss family,
and delegates boundary validation to `QuadratureDispatch._decode_boundary`.

# Arguments
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Problem dimensionality.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- `nothing`

# Errors
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Throws if `rule` is not a supported Gauss-family rule.
- Throws if `boundary` is invalid.

# Notes
- This helper centralizes the shared validation logic for the public and
  internal Gauss refinement routines.
"""
@inline function _require_gauss_inputs(
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol,
)::Nothing
    (N >= 1)   || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")
    _require_gauss_rule(rule)
    QuadratureDispatch._decode_boundary(boundary)
    return nothing
end

"""
    _quadrature_value_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol,
    ) -> Float64

Evaluate the Gauss-family quadrature approximation of `f` on `[a, b]^dim`.

# Function description
This helper validates the input configuration and then calls
`QuadratureDispatch.quadrature` to compute the quadrature approximation using
the requested Gauss-family rule, boundary condition, subdivision count, and
dimensionality. The result is converted to `Float64` before being returned.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of subdivisions or composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- `Float64`:
  The quadrature value produced by the Gauss-family backend.

# Errors
- Propagates validation errors from [`_require_gauss_inputs`](@ref).
- Propagates errors from `QuadratureDispatch.quadrature`.

# Notes
- This helper performs no derivative-based work.
- It is used internally by the refinement-based Gauss error estimator.
"""
@inline function _quadrature_value_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol,
)::Float64
    _require_gauss_inputs(N, dim, rule, boundary)

    q = QuadratureDispatch.quadrature(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary,
    )

    return float(q)
end

"""
    _estimate_by_refinement_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol,
    )

Estimate the Gauss-family quadrature error by comparing coarse and refined
quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
Gauss-family quadrature rules. It computes

- a coarse quadrature value using `N` subdivisions, and
- a refined quadrature value using `2N` subdivisions,

then forms the refinement difference

```julia
diff = q_fine - q_coarse.
```

The absolute value of this difference is used as the effective error estimate,
while the returned named tuple also records the signed difference, mesh sizes,
and both quadrature evaluations.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Coarse subdivision count.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- `NamedTuple` with fields:
  - `method`      : method tag `:gauss_refinement_difference`
  - `rule`        : quadrature rule symbol
  - `boundary`    : boundary-condition symbol
  - `N_coarse`    : coarse subdivision count
  - `N_fine`      : refined subdivision count (`2N`)
  - `dim`         : dimensionality
  - `h_coarse`    : coarse mesh size
  - `h_fine`      : refined mesh size
  - `q_coarse`    : coarse quadrature value
  - `q_fine`      : refined quadrature value
  - `estimate`    : absolute refinement difference
  - `signed_diff` : signed refinement difference
  - `reference`   : refined quadrature value used as the internal reference

# Errors
- Propagates validation errors from [`_require_gauss_inputs`](@ref).
- Propagates errors from the quadrature-evaluation layer.

# Notes
- This estimator does not use derivatives, jets, or residual moments.
- The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
"""
function _estimate_by_refinement_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol,
)
    _require_gauss_inputs(N, dim, rule, boundary)

    aa = float(a)
    bb = float(b)

    h_coarse = (bb - aa) / N
    h_fine   = (bb - aa) / (2N)

    q_coarse = _quadrature_value_gauss(
        f, aa, bb, N, dim, rule, boundary
    )
    q_fine = _quadrature_value_gauss(
        f, aa, bb, 2N, dim, rule, boundary
    )

    diff = q_fine - q_coarse

    return (;
        method      = :gauss_refinement_difference,
        rule        = rule,
        boundary    = boundary,
        N_coarse    = N,
        N_fine      = 2N,
        dim         = dim,
        h_coarse    = h_coarse,
        h_fine      = h_fine,
        q_coarse    = q_coarse,
        q_fine      = q_fine,
        estimate    = abs(diff),
        signed_diff = diff,
        reference   = q_fine,
    )
end

"""
    error_estimate_1d_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol,
    )

Estimate the 1D Gauss-family quadrature error by refinement.

# Function description
This public helper is the 1D specialization of the Gauss-family
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_gauss`](@ref) with `dim = 1`.

# Arguments
- `f`:
  Scalar integrand callable accepting one positional argument.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Coarse subdivision count.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_gauss`](@ref),
  specialized to `dim = 1`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_1d_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol,
)
    return _estimate_by_refinement_gauss(
        f, a, b, N, 1, rule, boundary
    )
end

"""
    error_estimate_2d_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol,
    )

Estimate the 2D Gauss-family quadrature error by refinement.

# Function description
This public helper is the 2D specialization of the Gauss-family
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_gauss`](@ref) with `dim = 2`.

# Arguments
- `f`:
  Scalar integrand callable accepting two positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_gauss`](@ref),
  specialized to `dim = 2`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_2d_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol,
)
    return _estimate_by_refinement_gauss(
        f, a, b, N, 2, rule, boundary
    )
end

"""
    error_estimate_3d_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol,
    )

Estimate the 3D Gauss-family quadrature error by refinement.

# Function description
This public helper is the 3D specialization of the Gauss-family
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_gauss`](@ref) with `dim = 3`.

# Arguments
- `f`:
  Scalar integrand callable accepting three positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_gauss`](@ref),
  specialized to `dim = 3`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_3d_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol,
)
    return _estimate_by_refinement_gauss(
        f, a, b, N, 3, rule, boundary
    )
end

"""
    error_estimate_4d_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol,
    )

Estimate the 4D Gauss-family quadrature error by refinement.

# Function description
This public helper is the 4D specialization of the Gauss-family
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_gauss`](@ref) with `dim = 4`.

# Arguments
- `f`:
  Scalar integrand callable accepting four positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_gauss`](@ref),
  specialized to `dim = 4`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_4d_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol,
)
    return _estimate_by_refinement_gauss(
        f, a, b, N, 4, rule, boundary
    )
end

"""
    error_estimate_nd_gauss(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        dim::Int,
    )

Estimate the `dim`-dimensional Gauss-family quadrature error by refinement.

# Function description
This public helper is the generic `nd` specialization of the Gauss-family
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_gauss`](@ref) with the user-supplied `dim`.

# Arguments
- `f`:
  Scalar integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  Gauss-family quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `dim::Int`:
  Problem dimensionality.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_gauss`](@ref),
  specialized to the requested `dim`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_nd_gauss(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int,
)
    return _estimate_by_refinement_gauss(
        f, a, b, N, dim, rule, boundary
    )
end

"""
    error_estimate_gauss(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary,
    )

Unified public dispatcher for Gauss-family refinement-based error estimation.

# Function description
This function provides the main Gauss-family-specific entry point for the
refinement-based error-estimation layer. It dispatches to the matching
dimension-specific specialization:

- `dim == 1` → [`error_estimate_1d_gauss`](@ref)
- `dim == 2` → [`error_estimate_2d_gauss`](@ref)
- `dim == 3` → [`error_estimate_3d_gauss`](@ref)
- `dim == 4` → [`error_estimate_4d_gauss`](@ref)
- otherwise  → [`error_estimate_nd_gauss`](@ref)

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
  Gauss-family quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Returns
- The named tuple produced by the selected dimension-specific refinement
  estimator.

# Errors
- Throws if `rule` is not a supported Gauss-family rule.
- Throws if `boundary` is invalid.
- Propagates errors from the selected dimension-specific routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on derivative backends or jet construction.
"""
function error_estimate_gauss(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary,
)
    _require_gauss_rule(rule)
    QuadratureDispatch._decode_boundary(boundary)

    if dim == 1
        return error_estimate_1d_gauss(f, a, b, N, rule, boundary)
    elseif dim == 2
        return error_estimate_2d_gauss(f, a, b, N, rule, boundary)
    elseif dim == 3
        return error_estimate_3d_gauss(f, a, b, N, rule, boundary)
    elseif dim == 4
        return error_estimate_4d_gauss(f, a, b, N, rule, boundary)
    else
        return error_estimate_nd_gauss(
            f, a, b, N, rule, boundary;
            dim = dim,
        )
    end
end

end  # module ErrorGaussRefine