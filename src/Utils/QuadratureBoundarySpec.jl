# ============================================================================
# src/Utils/QuadratureBoundarySpec.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureBoundarySpec

Boundary-specification validation and normalization helpers for
`Maranatha.jl`.

# Module description
`Maranatha.Utils.QuadratureBoundarySpec` centralizes the handling of boundary
specifications that may be given either as a single symbol shared by all axes
or as a tuple/vector of per-axis boundary symbols.

# Main entry points
- [`_decode_boundary`](@ref)
- [`_boundary_at`](@ref)
- [`_validate_boundary_spec`](@ref)

# Notes
- These helpers are used by quadrature, error-estimation, configuration, and
  reporting code.
- Supported scalar boundary symbols are `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`,
  and `:LU_EXEX`.
"""
module QuadratureBoundarySpec

import ..JobLoggerTools

"""
    _decode_boundary(
        boundary::Symbol
    ) -> Tuple{Symbol,Symbol}

Decode a composite boundary selector into left/right local endpoint kinds.

# Function description
This helper maps the global boundary pattern into a pair of local endpoint tags
used by the Newton-Cotes composite assembly:

- `:closed` means the local block includes the endpoint node.
- `:opened` means the local block uses the shifted open-type construction.

Supported patterns are:

- `:LU_ININ` -> `(:closed, :closed)`
- `:LU_EXIN` -> `(:opened, :closed)`
- `:LU_INEX` -> `(:closed, :opened)`
- `:LU_EXEX` -> `(:opened, :opened)`

# Arguments
- `boundary`: Boundary pattern symbol.

# Returns
- `Tuple{Symbol,Symbol}`: `(Ltype, Rtype)`, each equal to `:closed` or `:opened`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is not one of
  `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, or `:LU_EXEX`.
"""
@inline function _decode_boundary(
    boundary::Symbol
)
    if boundary === :LU_ININ
        return (:closed, :closed)
    elseif boundary === :LU_EXIN
        return (:opened, :closed)
    elseif boundary === :LU_INEX
        return (:closed, :opened)
    elseif boundary === :LU_EXEX
        return (:opened, :opened)
    else
        JobLoggerTools.error_benji("boundary must be one of: :LU_ININ | :LU_EXIN | :LU_INEX | :LU_EXEX (got $boundary)")
    end
end

"""
    _boundary_at(boundary, d::Int, dim::Int) -> Symbol

Resolve the scalar boundary symbol used on axis `d`.

# Function description
If `boundary` is a scalar symbol, it is shared across all axes and returned
unchanged after validation. If `boundary` is a tuple or vector, this helper
checks that its length matches `dim`, validates `boundary[d]`, and returns the
axis-local symbol.

# Arguments
- `boundary`:
  Boundary specification, scalar or axis-wise.
- `d::Int`:
  Axis index to resolve.
- `dim::Int`:
  Problem dimension used when validating axis-wise specifications.

# Returns
- `Symbol`:
  Boundary symbol used on axis `d`.

# Errors
- Throws `ArgumentError` if an axis-wise specification has the wrong length or
  contains a non-symbol entry.
- Propagates invalid-boundary errors from [`_decode_boundary`](@ref).
"""
@inline function _boundary_at(
    boundary::Symbol,
    d::Int,
    dim::Int,
)::Symbol
    _decode_boundary(boundary)
    return boundary
end

@inline function _boundary_at(
    boundary::Tuple,
    d::Int,
    dim::Int,
)::Symbol
    length(boundary) == dim ||
        throw(ArgumentError("length(boundary) must equal dim"))

    bd = boundary[d]
    bd isa Symbol ||
        throw(ArgumentError("boundary[$d] must be a Symbol"))

    _decode_boundary(bd)
    return bd
end

@inline function _boundary_at(
    boundary::AbstractVector,
    d::Int,
    dim::Int,
)::Symbol
    length(boundary) == dim ||
        throw(ArgumentError("length(boundary) must equal dim"))

    bd = boundary[d]
    bd isa Symbol ||
        throw(ArgumentError("boundary[$d] must be a Symbol"))

    _decode_boundary(bd)
    return bd
end

"""
    _validate_boundary_spec(boundary, dim::Int) -> Nothing

Validate that `boundary` is a well-formed boundary specification for dimension
`dim`.

# Function description
This helper accepts either a scalar boundary symbol shared across all axes or a
tuple/vector of per-axis boundary symbols of length `dim`. Every resolved
axis-local entry is validated against [`_decode_boundary`](@ref).

# Arguments
- `boundary`:
  Boundary specification to validate.
- `dim::Int`:
  Problem dimension that the specification must be compatible with.

# Returns
- `nothing`

# Errors
- Throws `ArgumentError` if `dim < 1` or if an axis-wise specification has the
  wrong length or invalid element types.
- Propagates invalid-boundary errors from [`_decode_boundary`](@ref).
"""
@inline function _validate_boundary_spec(
    boundary,
    dim::Int,
)::Nothing
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    for d in 1:dim
        _boundary_at(boundary, d, dim)
    end
    return nothing
end

end  # module QuadratureBoundarySpec
