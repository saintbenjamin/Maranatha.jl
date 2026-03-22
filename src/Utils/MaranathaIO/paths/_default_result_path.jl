# ============================================================================
# src/Utils/MaranathaIO/paths/_default_result_path.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _default_result_path(
        save_dir,
        name_prefix,
        name_suffix,
        a,
        b,
        rule,
        boundary,
        Ns
    ) -> String

Construct the default output path for a saved result file.

# Function description
This helper builds a standard `.jld2` filename from the output directory,
dataset prefix, dataset suffix, quadrature rule, boundary-condition label,
and the explicit list of stored subdivision counts.

# Arguments
- `save_dir`: Output directory.
- `name_prefix`: User-facing prefix for the dataset.
- `name_suffix`: User-facing suffix for the dataset.
- `a`, `b`: Domain-bound specifications used when deciding whether the filename
  token should stay scalar or expand axis-by-axis.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary-condition specification.
- `Ns`: Collection of subdivision counts.

# Returns
- `String`: Full default output path.

# Errors
- No explicit validation is performed.

# Notes
- This helper only constructs a path string; it does not create files or
  directories.
- The generated filename includes the axis-aware rule/boundary token,
  `nsamples` suffix, and user-provided prefix/suffix.
"""
function _default_result_path(
    save_dir::AbstractString,
    name_prefix::AbstractString,
    name_suffix::AbstractString,
    a,
    b,
    rule,
    boundary,
    Ns
)
    ns_suffix = _build_nsamples_suffix(Ns)
    spec_str = _rule_boundary_filename_token(a, b, rule, boundary)

    return joinpath(
        save_dir,
        "result_$(name_prefix)_$(spec_str)_$(ns_suffix)_$(name_suffix).jld2"
    )
end
