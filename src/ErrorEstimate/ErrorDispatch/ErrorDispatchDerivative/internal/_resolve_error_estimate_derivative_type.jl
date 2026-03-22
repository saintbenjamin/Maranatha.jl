# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_resolve_error_estimate_derivative_type.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _resolve_error_estimate_derivative_type(
        a,
        b,
        dim::Int,
        real_type,
    ) -> DataType

Resolve the active scalar type used by the derivative-based error-estimation
dispatchers.

# Function description
This helper centralizes the type-resolution rule shared by
[`error_estimate_derivative_direct`](@ref) and
[`error_estimate_derivative_jet`](@ref).

If `real_type` is provided, it is used directly. Otherwise, axis-wise bounds
must match `dim`, and the active scalar type is inferred from all bound
components. For scalar bounds, `promote_type(typeof(a), typeof(b))` is used.

# Arguments
- `a`, `b`:
  Integration-bound specifications.
- `dim::Int`:
  Problem dimensionality.
- `real_type`:
  Optional scalar type override.

# Returns
- `DataType`:
  Active scalar type used by downstream derivative estimators.

# Errors
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
"""
function _resolve_error_estimate_derivative_type(
    a,
    b,
    dim::Int,
    real_type,
)
    if !isnothing(real_type)
        return real_type
    elseif a isa AbstractVector || a isa Tuple
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))
        return promote_type(map(typeof, a)..., map(typeof, b)...)
    else
        return promote_type(typeof(a), typeof(b))
    end
end
