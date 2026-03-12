# ============================================================================
# src/Utils/MaranathaIO.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MaranathaIO

import ..TOML
import ..JLD2

import ..Utils.JobLoggerTools

# ============================================================
# 1. Serialization helpers
# ============================================================

"""
    namedtuple_to_dict(
        res
    ) -> Dict{String,Any}

Convert a Maranatha result `NamedTuple` into a serialization-friendly dictionary.

# Function description
This helper converts the structured result object produced by the Maranatha
workflow into a plain dictionary composed of standard container and scalar
types.

The conversion preserves the full stored content, including:

- global integration metadata,
- step sizes and averages,
- detailed per-datapoint error descriptors.

The resulting structure is suitable for storage in formats such as `JLD2` or
other external representations that do not naturally preserve Julia
`NamedTuple` layout.

# Arguments
- `res`: Result object to convert.

# Returns
- `Dict{String,Any}`: Dictionary representation of the result.

# Errors
- No explicit validation is performed; field-access errors will propagate if
  `res` does not match the expected result layout.

# Notes
- Each entry of `res.err` is converted into a plain dictionary.
- This helper is intended to pair with [`dict_to_namedtuple`](@ref).
"""
function namedtuple_to_dict(
    res
)
    return Dict(
        "a"           => res.a,
        "b"           => res.b,
        "h"           => collect(float.(res.h)),
        "avg"         => collect(float.(res.avg)),
        "rule"        => String(res.rule),
        "boundary"    => String(res.boundary),
        "dim"         => Int(res.dim),
        "err_method"  => String(res.err_method),
        "nerr_terms"  => Int(res.nerr_terms),
        "fit_terms"   => Int(res.fit_terms),
        "ff_shift"    => Int(res.ff_shift),
        "use_threads" => Bool(res.use_threads),
        "err" => [
            Dict(
                "ks"          => collect(Int.(e.ks)),
                "coeffs"      => collect(float.(e.coeffs)),
                "derivatives" => collect(float.(e.derivatives)),
                "terms"       => collect(float.(e.terms)),
                "total"       => float(e.total),
                "center"      => e.center isa Tuple ? collect(float.(e.center)) : float(e.center),
                "h"           => float(e.h),
            ) for e in res.err
        ]
    )
end

"""
    dict_to_namedtuple(
        d
    ) -> NamedTuple

Reconstruct a Maranatha result `NamedTuple` from a serialized dictionary.

# Function description
This helper performs the inverse conversion of [`namedtuple_to_dict`](@ref).

It restores the internal result structure expected by downstream analysis code,
including:

- numeric vectors,
- symbolic rule / boundary metadata,
- the vector of per-datapoint error descriptors.

# Arguments
- `d`: Dictionary representation of a result object.

# Returns
- `NamedTuple`: Reconstructed Maranatha result object.

# Errors
- No explicit schema validation is performed.
- Missing-key or incompatible-type errors will propagate if `d` does not match
  the expected serialized layout.

# Notes
- The stored `center` field is reconstructed as either a scalar `Float64` or a
  tuple of `Float64`, depending on the serialized representation.
"""
function dict_to_namedtuple(
    d
)
    err = [
        (
            ks          = Vector{Int}(e["ks"]),
            coeffs      = Vector{Float64}(e["coeffs"]),
            derivatives = Vector{Float64}(e["derivatives"]),
            terms       = Vector{Float64}(e["terms"]),
            total       = Float64(e["total"]),
            center      = e["center"] isa AbstractVector ? Tuple(Float64.(e["center"])) : Float64(e["center"]),
            h           = Float64(e["h"]),
        ) for e in d["err"]
    ]

    return (
        a           = Float64(d["a"]),
        b           = Float64(d["b"]),
        h           = Vector{Float64}(d["h"]),
        avg         = Vector{Float64}(d["avg"]),
        err         = err,
        rule        = Symbol(d["rule"]),
        boundary    = Symbol(d["boundary"]),
        dim         = Int(d["dim"]),
        err_method  = Symbol(d["err_method"]),
        nerr_terms  = Int(d["nerr_terms"]),
        fit_terms   = Int(d["fit_terms"]),
        ff_shift    = Int(d["ff_shift"]),
        use_threads = Bool(d["use_threads"]),
    )
end


# ============================================================
# 2. TOML summary helpers
# ============================================================

"""
    generate_summary_dict(
        res
    ) -> Dict{String,Any}

Generate a human-readable summary dictionary for [`TOML`](https://toml.io/en/) export.

# Function description
This helper builds a simplified dictionary view of a Maranatha result for
inspection and diagnostics.

The summary includes:

- integration metadata,
- step sizes,
- quadrature estimates,
- total error estimates,
- full per-datapoint error decomposition.

Unlike [`namedtuple_to_dict`](@ref), this representation is intended mainly for
human-readable summary export rather than structured round-trip recovery.

# Arguments
- `res`: Result object to summarize.

# Returns
- `Dict{String,Any}`: Summary dictionary suitable for [`TOML`](https://toml.io/en/) printing.

# Errors
- No explicit validation is performed; field-access errors propagate if `res`
  does not match the expected result layout.

# Notes
- This summary is commonly written as a companion `.toml` file next to a `JLD2`
  result file.
"""
function generate_summary_dict(
    res
)
    return Dict(
        "a"           => float(res.a),
        "b"           => float(res.b),
        "dim"         => Int(res.dim),
        "rule"        => String(res.rule),
        "boundary"    => String(res.boundary),
        "err_method"  => String(res.err_method),
        "nerr_terms"  => Int(res.nerr_terms),
        "fit_terms"   => Int(res.fit_terms),
        "ff_shift"    => Int(res.ff_shift),
        "use_threads" => Bool(res.use_threads),
        "h"           => collect(float.(res.h)),
        "avg"         => collect(float.(res.avg)),
        "err_total"   => [float(e.total) for e in res.err],
        "err" => [
            Dict(
                "ks"          => collect(Int.(e.ks)),
                "coeffs"      => collect(float.(e.coeffs)),
                "derivatives" => collect(float.(e.derivatives)),
                "terms"       => collect(float.(e.terms)),
                "total"       => float(e.total),
                "center"      => e.center isa Number ? float(e.center) : collect(float.(e.center)),
                "h"           => float(e.h),
            ) for e in res.err
        ]
    )
end


# ============================================================
# 3. Save / load API
# ============================================================

"""
    save_datapoint_results(
        path, 
        res; 
        write_summary=true
    ) -> String

Save a Maranatha result object to disk.

# Function description
This routine serializes a result object into a `.jld2` file after first
converting it into a plain dictionary representation via
[`namedtuple_to_dict`](@ref).

Optionally, it also writes a human-readable [`TOML`](https://toml.io/en/) summary generated by
[`generate_summary_dict`](@ref).

# Arguments
- `path`: Destination `.jld2` path.
- `res`: Result object to save.

# Keyword arguments
- `write_summary`: Whether to also write a companion `.toml` summary file.

# Returns
- `String`: Path to the written `.jld2` file.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `path` does not end with
  `.jld2`.
- Propagates file-writing and serialization errors from `JLD2` / [`TOML`](https://toml.io/en/) output.

# Notes
- The `JLD2` dataset key is `"datapoint_results"`.
"""
function save_datapoint_results(
    path::AbstractString,
    res;
    write_summary::Bool = true,
)
    endswith(lowercase(path), ".jld2") || JobLoggerTools.error_benji(
        "save_datapoint_results expects a .jld2 path (got path=$path)"
    )

    d = namedtuple_to_dict(res)

    JLD2.jldsave(path; datapoint_results=d)

    if write_summary
        toml_path = replace(path, r"\.jld2$" => ".toml")
        open(toml_path, "w") do io
            TOML.print(io, generate_summary_dict(res))
        end
    end

    return path
end

"""
    load_datapoint_results(
        path
    ) -> NamedTuple

Load a previously saved Maranatha result from disk.

# Function description
This routine reads a `.jld2` file produced by [`save_datapoint_results`](@ref),
loads the stored dictionary payload, and reconstructs the internal result
structure via [`dict_to_namedtuple`](@ref).

# Arguments
- `path`: Path to a `.jld2` file containing a stored result.

# Returns
- `NamedTuple`: Reconstructed result object.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `path` does not end with
  `.jld2`.
- Propagates `JLD2` loading and reconstruction errors.

# Notes
- The file is expected to contain the dataset key `"datapoint_results"`.
"""
function load_datapoint_results(
    path::AbstractString
)
    endswith(lowercase(path), ".jld2") || JobLoggerTools.error_benji(
        "load_datapoint_results expects a .jld2 path (got path=$path)"
    )

    d = JLD2.load(path, "datapoint_results")
    return dict_to_namedtuple(d)
end

"""
    infer_nsamples(
        res;
        atol=1e-10
    ) -> Vector{Int}

Infer subdivision counts `N` from the stored step sizes `h`.

# Function description
This helper reconstructs the effective subdivision counts from the standard
relation

```math
N = \\frac{b-a}{h}.
```

It is useful when a result object stores only `h` values but the corresponding
integer sample counts are needed for inspection, diagnostics, or filename
construction.

# Arguments
- `res`: Result object containing `a`, `b`, and stored step sizes `h`.

# Keyword arguments
- `atol`: Absolute tolerance used when checking numerical consistency with an
  integer subdivision count.

# Returns
- `Vector{Int}`: Inferred subdivision counts.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if a stored step size is not
  numerically consistent with an integer `N`.

# Notes
- This helper assumes the standard Maranatha convention `h = (b-a)/N`.
"""
function infer_nsamples(
    res;
    atol::Float64 = 1e-10,
)
    L = float(res.b - res.a)
    Ns = Int[]

    for hi in res.h
        x = L / float(hi)
        Ni = round(Int, x)

        isapprox(x, Ni; atol=atol, rtol=0.0) || JobLoggerTools.error_benji(
            "Failed to infer integer N from h=$hi on interval [$(res.a), $(res.b)]"
        )

        push!(Ns, Ni)
    end

    return Ns
end

"""
    _build_nsamples_suffix(
        Ns
    ) -> String

Construct a filename-friendly suffix from subdivision counts.

# Function description
This internal helper converts a collection of subdivision counts into a compact
suffix of the form

    N_2_3_4_5

for use in default result filenames.

# Arguments
- `Ns`: Collection of subdivision counts.

# Returns
- `String`: Filename-friendly `N_...` suffix.

# Errors
- No explicit validation is performed.

# Notes
- This helper is intended for internal path-construction workflows.
"""
function _build_nsamples_suffix(
    Ns
)
    return "N_" * join(Int.(Ns), "_")
end

"""
    _default_result_path(
        save_dir,
        name_prefix,
        rule,
        boundary,
        Ns
    ) -> String

Construct the default output path for a saved result file.

# Function description
This helper builds a standard `.jld2` filename from the output directory,
user-facing prefix, rule label, boundary label, and explicit list of stored
subdivision counts.

# Arguments
- `save_dir`: Output directory.
- `name_prefix`: User-facing prefix for the dataset.
- `rule`: Quadrature rule label.
- `boundary`: Boundary-condition label.
- `Ns`: Collection of subdivision counts.

# Returns
- `String`: Full default output path.

# Errors
- No explicit validation is performed.

# Notes
- This helper only constructs a path string; it does not create files or
  directories.
"""
function _default_result_path(
    save_dir::AbstractString,
    name_prefix::AbstractString,
    rule,
    boundary,
    Ns
)
    ns_suffix = _build_nsamples_suffix(Ns)

    return joinpath(
        save_dir,
        "result_$(name_prefix)_$(rule)_$(boundary)_$(ns_suffix).jld2"
    )
end

"""
    _assert_same_result_shape(
        results
    ) -> Nothing

Check that multiple result objects are mutually compatible for merging.

# Function description
This helper verifies that a collection of result objects shares the same global
integration metadata before datapoint arrays are merged.

The merge is considered valid only when all inputs correspond to the same
overall computational setup and differ only in their sampled datapoints.

# Arguments
- `results`: Collection of result objects to compare.

# Returns
- `nothing`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two results
  are supplied.
- Throws if any required metadata field differs between result objects.

# Notes
- This helper checks metadata compatibility only; it cannot prove semantic
  identity of the original integrand.
"""
function _assert_same_result_shape(
    results
)
    length(results) >= 2 || JobLoggerTools.error_benji(
        "_assert_same_result_shape requires at least 2 results"
    )

    ref = results[1]

    for (k, res) in enumerate(results[2:end])
        i = k + 1

        res.a == ref.a || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: a differs ($(res.a) != $(ref.a))"
        )
        res.b == ref.b || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: b differs ($(res.b) != $(ref.b))"
        )
        res.dim == ref.dim || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: dim differs ($(res.dim) != $(ref.dim))"
        )
        res.rule == ref.rule || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: rule differs ($(res.rule) != $(ref.rule))"
        )
        res.boundary == ref.boundary || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: boundary differs ($(res.boundary) != $(ref.boundary))"
        )
        res.err_method == ref.err_method || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: err_method differs ($(res.err_method) != $(ref.err_method))"
        )
        res.nerr_terms == ref.nerr_terms || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: nerr_terms differs ($(res.nerr_terms) != $(ref.nerr_terms))"
        )
        res.fit_terms == ref.fit_terms || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: fit_terms differs ($(res.fit_terms) != $(ref.fit_terms))"
        )
        res.ff_shift == ref.ff_shift || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: ff_shift differs ($(res.ff_shift) != $(ref.ff_shift))"
        )
        res.use_threads == ref.use_threads || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: use_threads differs ($(res.use_threads) != $(ref.use_threads))"
        )
    end

    return nothing
end

"""
    _assert_no_duplicate_h(
        hs;
        atol=1e-12
    ) -> Nothing

Check that a collection of step sizes contains no duplicate values.

# Function description
This helper sorts the supplied step sizes and checks adjacent values for
numerical duplication within the specified absolute tolerance.

It is used as a merge-safety check when combining datapoint result blocks.

# Arguments
- `hs`: Collection of step sizes.
- `atol`: Absolute tolerance used for duplicate detection.

# Returns
- `nothing`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if a duplicate step size is
  detected within tolerance.

# Notes
- This helper is conservative by design and rejects ambiguous overlapping
  datapoints.
"""
function _assert_no_duplicate_h(
    hs;
    atol::Float64 = 1e-12,
)
    p = sortperm(hs)
    hs_sorted = hs[p]

    for i in 2:length(hs_sorted)
        isapprox(hs_sorted[i], hs_sorted[i-1]; atol=atol, rtol=0.0) && JobLoggerTools.error_benji(
            "Duplicate h detected during merge: h=$(hs_sorted[i])"
        )
    end

    return nothing
end

"""
    merge_datapoint_results(
        results...;
        sort_by_h=true,
        allow_duplicate_h=false
    ) -> NamedTuple

Merge multiple compatible datapoint result blocks into one result object.

# Function description
This routine concatenates the aligned datapoint arrays

- `h`
- `avg`
- `err`

from several compatible result objects, optionally checks for duplicate step
sizes, and optionally sorts the merged datapoints by descending `h`.

Global metadata fields are copied from the first input result after shape
compatibility has been verified.

# Arguments
- `results...`: Compatible result objects to merge.

# Keyword arguments
- `sort_by_h`: Whether to sort the merged datapoints by descending `h`.
- `allow_duplicate_h`: Whether duplicate `h` values are allowed.

# Returns
- `NamedTuple`: Merged result object.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two results
  are provided.
- Throws if result metadata is incompatible or if aligned datapoint arrays have
  inconsistent lengths.
- Throws if duplicate `h` values are found and `allow_duplicate_h == false`.

# Notes
- This helper performs metadata-based safety checks, not full semantic identity
  checks on the originating problem.
"""
function merge_datapoint_results(
    results...;
    sort_by_h::Bool = true,
    allow_duplicate_h::Bool = false,
)
    length(results) >= 2 || JobLoggerTools.error_benji(
        "merge_datapoint_results requires at least 2 result inputs"
    )

    result_list = collect(results)

    _assert_same_result_shape(result_list)

    h_all = Float64[]
    avg_all = Float64[]
    err_all = eltype(result_list[1].err)[]

    for (i, res) in enumerate(result_list)
        length(res.h) == length(res.avg) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(avg)"
        )
        length(res.h) == length(res.err) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(err)"
        )

        append!(h_all, Float64.(res.h))
        append!(avg_all, Float64.(res.avg))
        append!(err_all, res.err)
    end

    if !allow_duplicate_h
        _assert_no_duplicate_h(h_all)
    end

    if sort_by_h
        p = sortperm(h_all; rev=true)
        h_all = h_all[p]
        avg_all = avg_all[p]
        err_all = err_all[p]
    end

    ref = result_list[1]

    return (
        a           = ref.a,
        b           = ref.b,
        h           = h_all,
        avg         = avg_all,
        err         = err_all,
        rule        = ref.rule,
        boundary    = ref.boundary,
        dim         = ref.dim,
        err_method  = ref.err_method,
        nerr_terms  = ref.nerr_terms,
        fit_terms   = ref.fit_terms,
        ff_shift    = ref.ff_shift,
        use_threads = ref.use_threads,
    )
end

"""
    merge_datapoint_result_files(
        paths...;
        output_path,
        write_summary=true,
        sort_by_h=true,
        allow_duplicate_h=false,
        output_dir::String = ".",
        name_prefix::String = "merged",
    ) -> String

Load multiple saved result files, merge them, and write the combined result.

# Function description
This is a file-based convenience wrapper around
[`merge_datapoint_results`](@ref).

It loads each input `.jld2` file, merges the corresponding in-memory result
objects, determines the final output path, and saves the merged result back to
disk.

# Arguments
- `paths...`: Input result-file paths.

# Keyword arguments
- `output_path`: Destination `.jld2` file path, or `nothing` to auto-generate.
- `write_summary`: Whether to write a companion [`TOML`](https://toml.io/en/) summary.
- `sort_by_h`: Whether to sort merged datapoints by descending `h`.
- `allow_duplicate_h`: Whether duplicate `h` values are allowed.
- `output_dir::String`: Output directory used when generating a default path.
- `name_prefix::String`: Filename prefix used when generating a default path.

# Returns
- `String`: Output path of the written merged `.jld2` file.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two input
  files are supplied.
- Propagates loading, merging, path-construction, and saving errors.

# Notes
- The merged [`TOML`](https://toml.io/en/) summary is regenerated from the merged in-memory result.
"""
function merge_datapoint_result_files(
    paths::AbstractString...;
    output_path::Union{Nothing,AbstractString} = nothing,
    write_summary::Bool = true,
    sort_by_h::Bool = true,
    allow_duplicate_h::Bool = false,
    output_dir::String = ".",
    name_prefix::String = "merged",
)
    length(paths) >= 2 || JobLoggerTools.error_benji(
        "merge_datapoint_result_files requires at least 2 input files"
    )

    results = map(load_datapoint_results, paths)

    merged = merge_datapoint_results(
        results...;
        sort_by_h = sort_by_h,
        allow_duplicate_h = allow_duplicate_h,
    )

    final_output_path = if output_path === nothing
        Ns = infer_nsamples(merged)
        _default_result_path(
            output_dir,
            name_prefix,
            merged.rule,
            merged.boundary,
            Ns,
        )
    else
        output_path
    end

    save_datapoint_results(
        final_output_path,
        merged;
        write_summary = write_summary,
    )

    return final_output_path
end

"""
    drop_nsamples_from_result(
        res,
        Ns_to_drop;
        atol=1e-10
    ) -> NamedTuple

Remove selected subdivision counts from a result object.

# Function description
This helper reconstructs the stored subdivision counts from the step sizes via
[`infer_nsamples`](@ref), builds a keep-mask, and returns a filtered copy of
the input result with the corresponding datapoint-aligned arrays reduced
consistently.

# Arguments
- `res`: Result object to filter.
- `Ns_to_drop`: Collection of subdivision counts to remove.

# Keyword arguments
- `atol`: Absolute tolerance used when reconstructing subdivision counts.

# Returns
- `NamedTuple`: Filtered result object.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if filtering would remove
  all datapoints.
- Propagates reconstruction errors from [`infer_nsamples`](@ref).

# Notes
- Global metadata fields are copied unchanged from the original result.
"""
function drop_nsamples_from_result(
    res,
    Ns_to_drop;
    atol::Float64 = 1e-10,
)
    Ns_all = infer_nsamples(res; atol=atol)

    drop_set = Set(Int.(Ns_to_drop))

    keep_mask = [!(N in drop_set) for N in Ns_all]

    any(keep_mask) || JobLoggerTools.error_benji(
        "drop_nsamples_from_result would remove all datapoints."
    )

    h_new   = res.h[keep_mask]
    avg_new = res.avg[keep_mask]
    err_new = res.err[keep_mask]

    return (
        a           = res.a,
        b           = res.b,
        h           = h_new,
        avg         = avg_new,
        err         = err_new,
        rule        = res.rule,
        boundary    = res.boundary,
        dim         = res.dim,
        err_method  = res.err_method,
        nerr_terms  = res.nerr_terms,
        fit_terms   = res.fit_terms,
        ff_shift    = res.ff_shift,
        use_threads = res.use_threads,
    )
end

"""
    drop_nsamples_from_file(
        input_path::AbstractString,
        Ns_to_drop;
        output_path::Union{Nothing,AbstractString} = nothing,
        write_summary::Bool = true,
        atol::Float64 = 1e-10,
        output_dir::String = ".",
        name_prefix::String = "dropped",
    ) -> String

Remove selected subdivision counts from a saved result file.

# Function description
This is a file-level convenience wrapper around
[`drop_nsamples_from_result`](@ref).

It loads an existing result file, filters out datapoints corresponding to the
specified subdivision counts, determines the final output path, and writes the
filtered result back to disk.

# Arguments
- `input_path::AbstractString`: Source `.jld2` result file.
- `Ns_to_drop`: Collection of subdivision counts to remove.

# Keyword arguments
- `output_path::Union{Nothing,AbstractString}`: Destination `.jld2` file path,
  or `nothing` to auto-generate one.
- `write_summary::Bool`: Whether to write a companion [`TOML`](https://toml.io/en/) summary.
- `atol::Float64`: Absolute tolerance used when reconstructing subdivision counts.
- `output_dir::String`: Output directory used when generating a default path.
- `name_prefix::String`: Filename prefix used when generating a default path.

# Returns
- `String`: Output path of the written filtered `.jld2` file.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if filtering would remove
  all datapoints.
- Propagates loading, filtering, path-construction, and saving errors.

# Notes
- The filtered result preserves all global metadata from the original file.
"""
function drop_nsamples_from_file(
    input_path::AbstractString,
    Ns_to_drop;
    output_path::Union{Nothing,AbstractString} = nothing,
    write_summary::Bool = true,
    atol::Float64 = 1e-10,
    output_dir::String = ".",
    name_prefix::String = "dropped",
)
    res = load_datapoint_results(input_path)

    filtered = drop_nsamples_from_result(
        res,
        Ns_to_drop;
        atol = atol,
    )

    final_output_path = if output_path === nothing
        Ns = infer_nsamples(filtered; atol=atol)
        _default_result_path(
            output_dir,
            name_prefix,
            filtered.rule,
            filtered.boundary,
            Ns,
        )
    else
        output_path
    end

    final_output_path = if output_path === nothing
        Ns = infer_nsamples(filtered; atol=atol)
        _default_result_path(
            output_dir,
            name_prefix,
            filtered.rule,
            filtered.boundary,
            Ns,
        )
    else
        output_path
    end

    save_datapoint_results(
        final_output_path,
        filtered;
        write_summary = write_summary,
    )

    return final_output_path
end

end  # module MaranathaIO