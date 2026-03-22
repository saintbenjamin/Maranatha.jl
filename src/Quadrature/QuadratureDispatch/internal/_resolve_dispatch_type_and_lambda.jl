# ============================================================================
# src/Quadrature/QuadratureDispatch/internal/_resolve_dispatch_type_and_lambda.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _resolve_dispatch_type_and_lambda(
        a,
        b,
        λ,
        real_type,
    ) -> NamedTuple

Resolve the active scalar type and normalized `λ` value used by the public
quadrature dispatcher.

# Function description
This helper centralizes the scalar-type and optional-parameter preparation used
by [`quadrature`](@ref).

If `real_type` is provided, it is used directly as the active scalar type.
Otherwise, the helper preserves the current dispatcher behavior by using
`promote_type(typeof(a), typeof(b))`.

The optional parameter `λ` is then normalized into the active scalar type.
If `λ === nothing`, zero in the active scalar type is used.

# Arguments
- `a`, `b`:
  Integration-bound specifications forwarded from the public dispatcher.
- `λ`:
  Optional rule parameter supplied to the public dispatcher.
- `real_type`:
  Optional explicit scalar type override.

# Returns
- `NamedTuple`:
  A bundle with fields:
  - `T`: active scalar type
  - `λT`: normalized `λ` value in type `T`

# Errors
- Propagates conversion errors from `convert(T, λ)` if `λ` cannot be converted
  into the resolved scalar type.

# Notes
- This helper intentionally preserves the current dispatch-level type-resolution
  semantics so that this refactor remains behavior-conservative.
"""
function _resolve_dispatch_type_and_lambda(
    a,
    b,
    λ,
    real_type,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    return (;
        T = T,
        λT = λT,
    )
end
