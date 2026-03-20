# ============================================================================
# src/Documentation/Reporter/_build_convergence_summary_datapoints_basename.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_convergence_summary_datapoints_basename(
        name,
        rule,
        boundary,
        h_power,
        xscale,
        yscale,
    ) -> String

Construct a standardized basename for datapoints-only convergence summary outputs.

# Function description

This helper builds a filesystem-friendly basename encoding the datapoint-plot
configuration used in the summary, including:

- the file-safe report or dataset name,
- the horizontal power ``h^{p}``,
- the x-axis scaling mode,
- the y-axis scaling mode.

The resulting basename is intended for use when writing summary files such as
[``\\LaTeX``](https://www.latex-project.org/) fragments or Markdown reports.

# Arguments

- `name`: User-facing identifier or file-derived label.
- `rule`: Quadrature rule label. Accepted for interface symmetry with related
  naming helpers, but not currently embedded in the returned basename.
- `boundary`: Boundary-handling label. Accepted for interface symmetry with
  related naming helpers, but not currently embedded in the returned basename.
- `h_power`: Power used in the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale keyword.
- `yscale`: Vertical axis scale keyword.

# Returns

- `String`: A standardized basename for datapoints-only summary artifacts.

# Notes

- The input `name` is sanitized internally via [`DocUtils._split_report_name`](@ref) so
  that path-like strings or `.jld2` filenames can be used safely.
- The current basename emphasizes the datapoint-plot configuration rather than
  rule, boundary, or fit metadata.
"""
function _build_convergence_summary_datapoints_basename(
    name::AbstractString,
    rule,
    boundary,
    h_power,
    xscale,
    yscale,
)
    _, file_name = DocUtils._split_report_name(name)

    return "summary_$(file_name)" *
           "_hpow_$(h_power)_$(String(xscale))_$(String(yscale))"
end