# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_resolve_threaded_subgrid_type_and_lambda.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _resolve_threaded_subgrid_type_and_lambda(
        a,
        b,
        dim::Int,
        λ,
        real_type,
    ) -> NamedTuple

Resolve the active scalar type and normalized `λ` value used by the threaded
subgrid backend entry point.

# Function description
This helper centralizes the scalar-type and optional-parameter preparation used
by [`quadrature_threaded_subgrid`](@ref).

If `real_type` is provided, it is used directly as the active scalar type.
Otherwise, the helper preserves the current backend behavior by:

- validating axis-wise bound lengths when `a` / `b` are tuple- or vector-like,
- inferring the scalar type from all bound components in that case,
- or using `promote_type(typeof(a), typeof(b))` for scalar bounds.

The optional parameter `λ` is then normalized into the active scalar type.
If `λ === nothing`, zero in the active scalar type is used.

# Arguments
- `a`, `b`:
  Integration-bound specifications.
- `dim::Int`:
  Problem dimensionality.
- `λ`:
  Optional rule parameter.
- `real_type`:
  Optional explicit scalar type override.

# Returns
- `NamedTuple`:
  A bundle with fields:
  - `T`: active scalar type
  - `λT`: normalized `λ` value in type `T`

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Throws `ArgumentError` if axis-wise bounds are supplied while inferring the
  scalar type and `length(a) != dim` or `length(b) != dim`.
- Propagates conversion errors from `convert(T, λ)`.

# Notes
- This helper intentionally preserves the current threaded-subgrid entry
  semantics.
"""
function _resolve_threaded_subgrid_type_and_lambda(
    a,
    b,
    dim::Int,
    λ,
    real_type,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    return (;
        T = T,
        λT = λT,
    )
end
