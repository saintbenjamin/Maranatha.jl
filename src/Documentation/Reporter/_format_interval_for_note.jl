# ============================================================================
# src/Documentation/Reporter/_format_interval_for_note.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _format_interval_for_note(a, b) -> String

Format an integration interval (possibly multi-dimensional) as a compact text string.

# Function description

This helper converts the domain specification given by `a` and `b` into a
human-readable interval representation suitable for inclusion in reports,
notes, captions, or table entries.

Supported cases:

- **Scalar domain**: returns a single interval `(a, b)`.
- **Rectangular domain**: when `a` and `b` are tuple- or vector-like objects
  of equal length, returns a comma-separated list of per-axis intervals:

  `(a₁, b₁), (a₂, b₂), …`

The function performs strict consistency checks to prevent ambiguous formatting.

# Arguments

- `a`, `b`: Domain endpoints.

  These must be either:

  - both scalars, or
  - both tuple/vector-like objects of equal length.

# Returns

- `String`: A formatted interval description.

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) if:

  - one endpoint is scalar and the other is tuple/vector-like, or
  - tuple/vector endpoints have mismatched lengths.

# Notes

- This routine performs formatting only; it does not validate numerical
  ordering or interpret domain semantics.
- The output is intended for textual presentation, not for programmatic parsing.
"""
@inline function _format_interval_for_note(a, b)
    a_is_multi = a isa Tuple || a isa AbstractVector
    b_is_multi = b isa Tuple || b isa AbstractVector

    if a_is_multi != b_is_multi
        JobLoggerTools.error_benji(
            "Interval-format mismatch: `a` and `b` must both be scalar or both be tuple/vector-like."
        )
    end

    if !(a_is_multi || b_is_multi)
        return "($(a), $(b))"
    end

    length(a) == length(b) || JobLoggerTools.error_benji(
        "Interval-format mismatch: length(a) != length(b)."
    )

    parts = ["($(a[i]), $(b[i]))" for i in eachindex(a)]
    return join(parts, ", ")
end