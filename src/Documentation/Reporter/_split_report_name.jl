# ============================================================================
# src/Documentation/Reporter/_split_report_name.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _split_report_name(name::AbstractString) -> Tuple{String,String}

Split a user-supplied report name into display and file-safe components.

# Function description

This helper separates a report identifier into two related forms:

1. `display_name`: the original string, preserved for human-readable use in
   report titles or captions,
2. `file_name`: a sanitized basename suitable for filesystem output.

It is especially useful when users pass a full file path, such as a stored
`.jld2` result file, as the `name` argument to reporting functions.

# Arguments

- `name::AbstractString`: Original report name, identifier, or file path.

# Returns

- `Tuple{String,String}`:
  A pair `(display_name, file_name)` where:
  - `display_name` preserves the original string,
  - `file_name` is reduced to `basename(name)` with a trailing `.jld2`
    suffix removed when present.

# Notes

- This helper prevents accidental insertion of directory separators into
  generated output filenames.
- It is intended for internal use by reporting and plotting helpers that need
  to distinguish between display labels and filesystem-safe basenames.
"""
function _split_report_name(name::AbstractString)
    display_name = String(name)
    file_name = replace(basename(String(name)), r"\.jld2$" => "")
    return display_name, file_name
end