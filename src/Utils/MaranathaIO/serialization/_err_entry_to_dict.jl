# ============================================================================
# src/Utils/MaranathaIO/serialization/_err_entry_to_dict.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _err_entry_to_dict(e) -> Dict{String,Any}

Convert a single internal error-entry object into a serialization-friendly dictionary.

# Function description
This helper normalizes one element of `res.err` into a plain dictionary composed
of standard scalar and container types suitable for storage in `JLD2`, `TOML`,
or related external formats.

This conversion supports both scalar-step and axis-wise-step error entries.
Fields such as `:center`, `:h`, `:h_coarse`, and `:h_fine` are preserved through
`_storable_domain_value`, so rectangular-domain metadata can be serialized
without losing per-axis structure.

It currently supports two error-entry layouts:

- residual-based error objects exposing fields such as `:ks`, `:coeffs`,
  `:derivatives`, `:terms`, and `:total`, and
- refinement-based error objects exposing fields such as `:method`,
  `:N_coarse`, `:N_fine`, and `:estimate`.

# Arguments
- `e`:
  One internal error-entry object, typically an element of `res.err`.

# Returns
- `Dict{String,Any}`:
  Serialization-friendly dictionary representation of the error entry.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `e` does not match any supported
  error-entry structure.

# Notes
- Residual-based entries are currently tagged with `"err_format" => "derivative"`.
- Refinement-based entries are currently tagged with `"err_format" => "refinement"`.
- Residual-based entries are serialized in the legacy flat layout. Any
  axis-wise decomposition stored in a `per_axis` field is not currently
  preserved by this helper.
"""
function _err_entry_to_dict(e)
    if hasproperty(e, :ks)
        return Dict(
            "err_format"   => "derivative",
            "ks"           => collect(Int.(e.ks)),
            "coeffs"       => collect(e.coeffs),
            "derivatives"  => collect(e.derivatives),
            "terms"        => collect(e.terms),
            "total"        => e.total,
            "center"       => _storable_domain_value(e.center),
            "h"            => _storable_domain_value(e.h),
        )
    elseif hasproperty(e, :estimate)
        return Dict(
            "err_format"   => "refinement",
            "method"       => String(e.method),
            "rule"         => _storable_rule_value(e.rule),
            "boundary"     => _storable_boundary_value(e.boundary),
            "N_coarse"     => Int(e.N_coarse),
            "N_fine"       => Int(e.N_fine),
            "dim"          => Int(e.dim),
            "h_coarse"     => _storable_domain_value(e.h_coarse),
            "h_fine"       => _storable_domain_value(e.h_fine),
            "q_coarse"     => e.q_coarse,
            "q_fine"       => e.q_fine,
            "estimate"     => e.estimate,
            "signed_diff"  => e.signed_diff,
            "reference"    => e.reference,
        )
    else
        JobLoggerTools.error_benji(
            "Unsupported error entry format during serialization. " *
            "Expected derivative or refinement structure."
        )
    end
end
