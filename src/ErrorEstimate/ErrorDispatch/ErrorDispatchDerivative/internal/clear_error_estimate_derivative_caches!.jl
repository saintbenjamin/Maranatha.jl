# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/clear_error_estimate_derivative_caches!.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    clear_error_estimate_derivative_caches!() -> Nothing

Clear all global caches used by the error-estimation layer.

# Function description
This helper empties the residual-model cache, derivative-value cache, and
derivative-jet cache, then prints the resulting cache sizes through
[`JobLoggerTools.println_benji`](@ref).

This is useful when:

- benchmarking cache behavior,
- forcing a clean recomputation,
- debugging stale cache contents,
- resetting state between large runs.

# Arguments
- None.

# Returns
- `nothing`

# Errors
- This helper does not throw on normal use; dictionary mutation failures are
  propagated if they occur.

# Side effects
- Mutates the following global caches:
  - [`_RES_MODEL_CACHE`](@ref)
  - [`_NTH_DERIV_CACHE`](@ref)
  - [`_DERIV_JET_CACHE`](@ref)

# Notes
- The printed sizes should normally all be zero immediately after this call.
"""
function clear_error_estimate_derivative_caches!()
    empty!(_RES_MODEL_CACHE)
    empty!(_NTH_DERIV_CACHE)
    empty!(_DERIV_JET_CACHE)
    return nothing
end
