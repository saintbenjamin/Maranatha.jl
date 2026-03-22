# ============================================================================
# src/Runner/internal/_resolve_real_type.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _resolve_real_type(
        real_type,
        use_cuda::Bool,
    )

Resolve the active scalar type for a runner invocation.

# Function description
This helper selects the scalar type used internally by the runner. If
`real_type` is `nothing`, the default type `Float64` is used. Otherwise, the
supplied type-like object is returned unchanged.

When CUDA execution is requested, this helper also enforces the current CUDA
backend restriction that only `Float32` and `Float64` are supported.

# Arguments
- `real_type`:
  Requested computation scalar type, or `nothing` to use the default type.
- `use_cuda::Bool`:
  Whether CUDA execution was requested for the current run.

# Returns
- Active scalar type used by the runner.

# Errors
- Throws `ArgumentError` if `use_cuda = true` and the resolved type is neither
  `Float32` nor `Float64`.

# Notes
- This helper resolves only the scalar type object. Domain conversion and
  integrand evaluation are handled elsewhere.
"""
@inline function _resolve_real_type(
    real_type,
    use_cuda::Bool,
)
    T = isnothing(real_type) ? Float64 : real_type

    if use_cuda && !(T === Float32 || T === Float64)
        throw(ArgumentError(
            "CUDA mode currently supports only Float32 or Float64 real_type (got $(T))."
        ))
    end

    return T
end