# ============================================================================
# src/Utils/MaranathaIO/serialization/_dict_to_err_entry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _dict_to_err_entry(e)

Reconstruct a single internal error-entry object from a serialized dictionary.

# Function description
This helper performs the inverse conversion of [`_err_entry_to_dict`](@ref).
It reads the stored `"err_format"` tag and rebuilds the corresponding internal
error-entry structure expected by downstream fitting, plotting, and reporting
code.

Currently supported serialized formats are:

- `"derivative"` for derivative-based error entries, and
- `"refinement"` for refinement-based error entries.

# Arguments
- `e`:
  Serialized dictionary representation of one error entry.

# Returns
- A reconstructed internal error-entry `NamedTuple`.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if the stored `"err_format"` value
  is unsupported.

# Notes
- The reconstructed structure is intended to match the field layout expected by
  downstream Maranatha workflows.
- Scalar and axis-wise geometric fields are restored through
  [`_restore_domain_value`](@ref), so rectangular-domain error metadata is
  reconstructed as tuples when appropriate.
- Residual-based entries are reconstructed in the legacy flat layout and do
  not currently restore a `per_axis` decomposition.
"""
function _dict_to_err_entry(e)
    fmt = get(e, "err_format", "refinement")

    if fmt == "derivative"
        coeffs = collect(e["coeffs"])
        T = isempty(coeffs) ? Float64 : eltype(coeffs)

        return (
            ks          = Vector{Int}(e["ks"]),
            coeffs      = Vector{T}(coeffs),
            derivatives = Vector{T}(e["derivatives"]),
            terms       = Vector{T}(e["terms"]),
            total       = convert(T, e["total"]),
            center      = _restore_domain_value(e["center"], T),
            h           = _restore_domain_value(e["h"], T),
        )
    elseif fmt == "refinement"
        T = typeof(e["estimate"])
        return (
            method      = Symbol(e["method"]),
            rule        = _restore_rule_value(e["rule"]),
            boundary    = _restore_boundary_value(e["boundary"]),
            N_coarse    = Int(e["N_coarse"]),
            N_fine      = Int(e["N_fine"]),
            dim         = Int(e["dim"]),
            h_coarse    = _restore_domain_value(e["h_coarse"], T),
            h_fine      = _restore_domain_value(e["h_fine"], T),
            q_coarse    = convert(T, e["q_coarse"]),
            q_fine      = convert(T, e["q_fine"]),
            estimate    = convert(T, e["estimate"]),
            signed_diff = convert(T, e["signed_diff"]),
            reference   = convert(T, e["reference"]),
        )
    else
        JobLoggerTools.error_benji(
            "Unsupported err_format during deserialization: err_format=$(fmt)"
        )
    end
end
