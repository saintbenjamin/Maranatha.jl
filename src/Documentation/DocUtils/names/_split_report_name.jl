# ============================================================================
# src/Documentation/DocUtils/names/_split_report_name.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _split_report_name(
        name::AbstractString
    ) -> Tuple{String,String}

Split a user-supplied report identifier into a display name and a
filesystem-safe base name.

# Function description

This helper normalizes a report name for two distinct purposes:

1. `display_name`: the original string converted to `String`, preserved for
   human-facing contexts such as titles, captions, or annotations.
2. `file_name`: a sanitized basename suitable for filesystem output,
   derived from the input path with any trailing `.jld2` suffix removed.

This is particularly useful when plotting or reporting routines accept either
a plain identifier or a path-like string (e.g., a saved result file) and need
a clean filename stem for generated artifacts.

# Arguments

- `name::AbstractString`:
  Report name, identifier, or path-like string.

# Returns

- `Tuple{String,String}`:
  A pair `(display_name, file_name)` where:

  - `display_name` preserves the full input content,
  - `file_name` is computed as `basename(String(name))` with a trailing
    `.jld2` suffix removed when present.

# Notes

- `display_name` is intended for human-readable output.
- `file_name` avoids directory separators and unsafe path components.
- Only a trailing `.jld2` extension is stripped; other suffixes are preserved.
- This helper is intended for internal use by reporting and plotting utilities.
"""
function _split_report_name(
    name::AbstractString
)
    display_name = String(name)
    file_name = replace(basename(String(name)), r"\.jld2$" => "")
    return display_name, file_name
end
