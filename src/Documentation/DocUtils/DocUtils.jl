# ============================================================================
# src/Documentation/DocUtils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DocUtils

Shared documentation-output helpers for `Maranatha.jl`.

# Module description
`Maranatha.Documentation.DocUtils` contains small helper functions reused by
plotting and reporting code. These utilities focus on name normalization and
compact axis-wise filename tokens derived from domain, rule, and boundary
metadata.

# Main entry points
- [`_split_report_name`](@ref)
- [`_rule_boundary_filename_token`](@ref)
"""
module DocUtils

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

"""
    _report_name_is_multi(x) -> Bool

Return `true` if `x` is treated as an axis-wise specification for report-name
construction.

# Function description
This helper classifies tuple and vector values as multi-axis specifications and
all other values as shared scalar specifications.

# Arguments
- `x`: Candidate report-metadata value.

# Returns
- `Bool`: `true` for tuple/vector input, `false` otherwise.

# Errors
- No explicit validation is performed.
"""
@inline _report_name_is_multi(x) = x isa Tuple || x isa AbstractVector

"""
    _report_name_cfg_dim(a, b, rule, boundary) -> Int

Infer the effective axis count used when constructing report filename tokens.

# Function description
This helper inspects domain bounds together with rule and boundary metadata and
returns the unique common axis count implied by any axis-wise inputs. Scalar
shared inputs contribute no axis count. If all inputs are scalar-like, the
returned dimension is `1`.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `Int`: Effective dimension used for filename-token expansion.

# Errors
- Throws if `a` and `b` mix scalar and collection styles.
- Throws if axis-wise inputs imply inconsistent dimensions.
"""
@inline function _report_name_cfg_dim(a, b, rule, boundary)::Int
    a_multi = _report_name_is_multi(a)
    b_multi = _report_name_is_multi(b)

    if a_multi != b_multi
        error("Filename-spec mismatch: `a` and `b` must both be scalar or both be tuple/vector-like.")
    end

    dims = Int[]

    if a_multi
        push!(dims, length(a))
        push!(dims, length(b))
    end
    _report_name_is_multi(rule)     && push!(dims, length(rule))
    _report_name_is_multi(boundary) && push!(dims, length(boundary))

    isempty(dims) && return 1

    dim = first(dims)
    all(==(dim), dims) || error(
        "Filename-spec mismatch: inconsistent axis counts across domain/rule/boundary."
    )

    return dim
end

"""
    _report_name_cfg_at(x, d::Int, dim::Int)

Resolve the value of one report-metadata specification on axis `d`.

# Function description
Scalar inputs are treated as shared values and returned unchanged. Tuple/vector
inputs are validated against `dim` and indexed at axis `d`.

# Arguments
- `x`: Scalar or axis-wise report-metadata specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise inputs.

# Returns
- The scalar shared value or the axis-local entry `x[d]`.

# Errors
- Throws if an axis-wise specification has length different from `dim`.
"""
@inline function _report_name_cfg_at(x, d::Int, dim::Int)
    if _report_name_is_multi(x)
        length(x) == dim || error("Filename-spec mismatch: spec length must equal dim.")
        return x[d]
    end
    return x
end

"""
    _rule_boundary_filename_token(a, b, rule, boundary) -> String

Construct the compact rule/boundary token used in report and figure filenames.

# Function description
If all inputs are scalar-like, this helper returns a compact token of the form
`"<rule>_<boundary>"`. If any input is axis-wise, it expands the result into an
axis-tagged token of the form
`"1_<rule1>_<boundary1>_2_<rule2>_<boundary2>_..."`.

# Arguments
- `a`, `b`: Domain-bound specifications used only to infer whether the run is
  scalar or axis-wise.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `String`: Filename-friendly rule/boundary token.

# Errors
- Propagates dimensional-consistency errors from [`_report_name_cfg_dim`](@ref)
  and [`_report_name_cfg_at`](@ref).

# Notes
- Domain values themselves are not embedded in the returned token.
"""
@inline function _rule_boundary_filename_token(a, b, rule, boundary)::String
    dim = _report_name_cfg_dim(a, b, rule, boundary)

    if dim == 1
        return "$(string(rule))_$(string(boundary))"
    end

    parts = String[]
    for d in 1:dim
        rd = _report_name_cfg_at(rule, d, dim)
        bd = _report_name_cfg_at(boundary, d, dim)
        push!(parts, string(d))
        push!(parts, string(rd))
        push!(parts, string(bd))
    end

    return join(parts, "_")
end

end  # module DocUtils
