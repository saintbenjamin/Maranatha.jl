# ============================================================================
# src/Runner/internal/_maybe_save_result.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _maybe_save_result(
        result,
        save_path::Union{Nothing,AbstractString},
        name_prefix::String,
        write_summary::Bool,
    ) -> Union{Nothing,String}

Optionally persist a completed runner result to disk.

# Function description
This helper encapsulates the runner's optional save step. If `save_path` is
`nothing`, the helper performs no action and returns `nothing`.

Otherwise, it normalizes the output directory, creates it if needed, builds the
standard result filename from the stored run metadata, writes the JLD2 result
via [`MaranathaIO.save_datapoint_results`](@ref), and optionally writes the
paired summary artifact according to `write_summary`.

# Arguments
- `result`:
  Completed runner result `NamedTuple`.
- `save_path::Union{Nothing,AbstractString}`:
  Output directory or `nothing` to disable saving.
- `name_prefix::String`:
  User-facing filename prefix.
- `write_summary::Bool`:
  Whether summary sidecar output should also be written.

# Returns
- `nothing` if saving is disabled.
- Absolute path of the written JLD2 file if saving is performed.

# Errors
- Propagates path-creation and file-writing errors from `mkpath` and
  [`MaranathaIO.save_datapoint_results`](@ref).
- Propagates filename-token construction errors from
  [`MaranathaIO._rule_boundary_filename_token`](@ref).

# Notes
- The filename embeds the normalized rule/boundary token derived from
  `result.a`, `result.b`, `result.rule`, and `result.boundary`.
"""
function _maybe_save_result(
    result,
    save_path::Union{Nothing,AbstractString},
    name_prefix::String,
    write_summary::Bool,
)
    save_path === nothing && return nothing

    save_path_abs = isabspath(save_path) ? save_path : joinpath(pwd(), save_path)
    mkpath(save_path_abs)

    Nstr = join(sort(result.nsamples), "_")
    spec_str = MaranathaIO._rule_boundary_filename_token(
        result.a,
        result.b,
        result.rule,
        result.boundary,
    )

    save_jld2_path = joinpath(
        save_path_abs,
        "result_$(name_prefix)_$(spec_str)_N_$(Nstr).jld2"
    )

    MaranathaIO.save_datapoint_results(
        save_jld2_path,
        result;
        write_summary = write_summary,
    )

    return save_jld2_path
end