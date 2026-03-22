# ============================================================================
# src/Utils/MaranathaIO/serialization/_err_entry_total.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _err_entry_total(e) -> Real

Extract the stored scalar error quantity from an internal error-entry object.

# Function description
This helper provides a unified scalar accessor across the currently supported
error-entry layouts.

It supports:

- derivative-style error entries exposing a `:total` field, and
- refinement-style error entries exposing an `:estimate` field.

The stored scalar value is returned as-is; this helper does not convert the
result to `Float64` and does not apply `abs(...)`.

# Arguments
- `e`:
  One internal error-entry object.

# Returns
- `Real`:
  Stored scalar error quantity associated with the entry.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `e` does not expose either
  `:total` or `:estimate`.

# Notes
- This helper is mainly used for summary export and human-readable diagnostics.
"""
function _err_entry_total(e)
    if hasproperty(e, :total)
        return e.total
    elseif hasproperty(e, :estimate)
        return e.estimate
    else
        JobLoggerTools.error_benji(
            "Unsupported error entry format while extracting total-like quantity."
        )
    end
end
