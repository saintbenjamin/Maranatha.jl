# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchCUDA/internal/_resolve_cuda_type_and_lambda.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _resolve_cuda_type_and_lambda(
        a,
        b,
        dim::Int,
        threads::Int,
        λ,
        real_type,
    ) -> NamedTuple

Resolve the active scalar type and normalized `λ` value used by the CUDA
quadrature backend entry point.

# Function description
This helper centralizes the scalar-type and optional-parameter preparation used
by [`quadrature_cuda`](@ref).

It preserves the current backend behavior by:

- validating `dim >= 1` and `threads >= 1`,
- honoring `real_type` directly when provided,
- otherwise inferring the scalar type from all bound components for axis-wise
  domains, or from scalar bounds for hypercube-style input,
- enforcing the current CUDA restriction that only `Float32` and `Float64`
  are accepted.

# Arguments
- `a`, `b`:
  Integration-bound specifications.
- `dim::Int`:
  Problem dimensionality.
- `threads::Int`:
  Requested CUDA threads per block.
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
- Throws `ArgumentError` if `threads < 1`.
- Throws `ArgumentError` if axis-wise bounds are supplied while inferring the
  scalar type and `length(a) != dim` or `length(b) != dim`.
- Throws `ArgumentError` if the active scalar type is not `Float32` or `Float64`.
- Propagates conversion errors from `convert(T, λ)`.
"""
function _resolve_cuda_type_and_lambda(
    a,
    b,
    dim::Int,
    threads::Int,
    λ,
    real_type,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    threads >= 1 || throw(ArgumentError("threads must be ≥ 1"))

    T = if !isnothing(real_type)
        real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        promote_type(typeof(a), typeof(b))
    end

    (T === Float32 || T === Float64) || throw(ArgumentError(
        "CUDA mode currently supports only Float32 or Float64 real_type (got $(T))."
    ))

    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    return (;
        T = T,
        λT = λT,
    )
end
