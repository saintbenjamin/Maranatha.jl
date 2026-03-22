# ============================================================================
# src/Utils/MaranathaIO/MaranathaIO.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module MaranathaIO

Serialization, summary-export, and result-file management helpers for
`Maranatha.jl`.

# Module description
`Maranatha.Utils.MaranathaIO` provides the I/O glue for saving, loading,
summarizing, merging, and filtering structured Maranatha result objects.

The module supports scalar and axis-wise domain / rule / boundary metadata and
encodes them in both serialized payloads and default filename conventions.

# Main entry points
- [`namedtuple_to_dict`](@ref)
- [`dict_to_namedtuple`](@ref)
- [`save_datapoint_results`](@ref)
- [`load_datapoint_results`](@ref)
- [`merge_datapoint_results`](@ref)
- [`merge_datapoint_result_files`](@ref)
- [`drop_nsamples_from_result`](@ref)
- [`drop_nsamples_from_file`](@ref)
"""
module MaranathaIO

import ..TOML
import ..JLD2

import ..Utils.JobLoggerTools

# ============================================================
# Internal helper includes
# ============================================================

# ------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------
include("serialization/_is_scalar_domain_value.jl")
include("serialization/_storable_domain_value.jl")
include("serialization/_storable_boundary_value.jl")
include("serialization/_storable_rule_value.jl")
include("serialization/_restore_rule_value.jl")
include("serialization/_restore_domain_value.jl")
include("serialization/_restore_boundary_value.jl")
include("serialization/_err_entry_to_dict.jl")
include("serialization/_dict_to_err_entry.jl")
include("serialization/_err_entry_total.jl")

# ------------------------------------------------------------
# Filename / path helpers
# ------------------------------------------------------------
include("paths/_filename_spec_is_multi.jl")
include("paths/_filename_spec_dim.jl")
include("paths/_filename_spec_at.jl")
include("paths/_rule_boundary_filename_token.jl")
include("paths/_to_axis_vector.jl")
include("paths/_build_nsamples_suffix.jl")
include("paths/_default_result_path.jl")

# ------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------
include("summary/_toml_safe.jl")

# ------------------------------------------------------------
# Merge / drop safety helpers
# ------------------------------------------------------------
include("merge_drop/_assert_same_result_shape.jl")
include("merge_drop/_assert_no_duplicate_h.jl")

# ============================================================
# Public serialization API
# ============================================================

"""
    namedtuple_to_dict(res) -> Dict{String,Any}

Convert a Maranatha result `NamedTuple` into a serialization-friendly dictionary.

# Function description
This helper converts the structured result object produced by the Maranatha
workflow into a plain dictionary composed of standard container and scalar
types.

The conversion preserves the stored result content needed by downstream
workflows, including:

- global integration metadata,
- scalarized step sizes in `res.h`,
- original per-axis step information in `res.tuple_h` when present,
- quadrature estimates, and
- detailed per-datapoint error descriptors in their serialized legacy form.

The resulting structure is suitable for storage in formats such as `JLD2` or
other external representations that do not naturally preserve Julia
`NamedTuple` layout.

The conversion supports both residual-based and refinement-based error-entry
objects stored in `res.err`, and preserves enough metadata to reconstruct them
later through [`dict_to_namedtuple`](@ref).

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
- Each entry of `res.err` may represent either a residual-based or a
  refinement-based error object.
- For rectangular-domain results, `tuple_h` preserves the original per-axis
  step tuples while `h` preserves the scalar fitting step sequence.
- Axis-wise `rule` and `boundary` specifications are preserved through the
  dedicated rule/boundary serialization helpers.
- Residual-based error entries are serialized in the flat legacy layout, so a
  `per_axis` field is not currently round-tripped.
"""
function namedtuple_to_dict(res)
    return Dict(
        "a"             => _storable_domain_value(res.a),
        "b"             => _storable_domain_value(res.b),
        "nsamples"      => hasproperty(res, :nsamples) ? collect(Int.(res.nsamples)) : Int[],
        "h"             => collect(res.h),
        "tuple_h"       => hasproperty(res, :tuple_h) ?
                           [_storable_domain_value(hi) for hi in res.tuple_h] :
                           [_storable_domain_value(hi) for hi in res.h],
        "avg"           => collect(res.avg),
        "rule"          => _storable_rule_value(res.rule),
        "boundary"      => _storable_boundary_value(res.boundary),
        "dim"           => Int(res.dim),
        "err_method"    => String(res.err_method),
        "nerr_terms"    => Int(res.nerr_terms),
        "fit_terms"     => Int(res.fit_terms),
        "ff_shift"      => Int(res.ff_shift),
        "use_error_jet" => Bool(res.use_error_jet),
        "use_cuda"      => getproperty(res, :use_cuda),
        "real_type"     => getproperty(res, :real_type),
        "err"           => [_err_entry_to_dict(e) for e in res.err],
    )
end

"""
    dict_to_namedtuple(d) -> NamedTuple

Reconstruct a Maranatha result `NamedTuple` from a serialized dictionary.

# Function description
This helper performs the inverse conversion of [`namedtuple_to_dict`](@ref).

It restores the internal result structure expected by downstream analysis code,
including:

- numeric vectors,
- symbolic rule / boundary metadata,
- scalarized step sizes in `h`,
- original per-axis step information in `tuple_h`,
- stored per-datapoint error descriptors, and
- subdivision counts in `nsamples`.

The reconstructed `err` field may therefore contain either derivative-based or
refinement-based error-entry objects, depending on the serialized content.

If the serialized payload omits `nsamples` or stores them as an empty
collection, they are reconstructed via [`infer_nsamples`](@ref).

# Arguments
- `d`: Dictionary representation of a result object.

# Returns
- `NamedTuple`: Reconstructed Maranatha result object with an `nsamples` field.

# Errors
- No explicit schema validation is performed.
- Missing-key or incompatible-type errors will propagate if `d` does not match
  the expected serialized layout.
- Propagates reconstruction errors from [`infer_nsamples`](@ref) when
  `nsamples` must be rebuilt from step-size information.

# Notes
- The numeric restoration type `T` is derived from `d["real_type"]` when
  available, otherwise from `eltype(d["avg"])`, and otherwise defaults to
  `Float64`.
- Stored geometric fields are reconstructed as either scalars in `T` or tuples
  with elements in `T`, depending on the serialized representation.
- Refinement-based error entries are reconstructed from their serialized
  `"err_format"` tag in the same unified `err` vector as derivative-based entries.
- If the serialized payload includes `tuple_h`, it is restored explicitly;
  otherwise, `h` is reused as a fallback for backward compatibility.
- `use_cuda` defaults to `false` and `real_type` defaults to `string(T)` when
  those keys are absent.
- Axis-wise `rule` and `boundary` specifications are restored as tuples of
  symbols when appropriate.
- Residual-based error entries are reconstructed in the legacy flat layout and
  therefore do not currently recover any original `per_axis` decomposition.
"""
function dict_to_namedtuple(d)
    err = [_dict_to_err_entry(e) for e in d["err"]]

    T = if haskey(d, "real_type")
        if d["real_type"] isa DataType
            d["real_type"]
        else
            getfield(Main, Symbol(d["real_type"]))
        end
    elseif !isempty(d["avg"])
        eltype(d["avg"])
    else
        Float64
    end

    tuple_h_restored = if haskey(d, "tuple_h")
        [_restore_domain_value(hi, T) for hi in d["tuple_h"]]
    else
        [_restore_domain_value(hi, T) for hi in d["h"]]
    end

    base_res = (
        a             = _restore_domain_value(d["a"], T),
        b             = _restore_domain_value(d["b"], T),
        h             = Vector{T}(d["h"]),
        tuple_h       = tuple_h_restored,
        avg           = Vector{T}(d["avg"]),
        err           = err,
        rule          = _restore_rule_value(d["rule"]),
        boundary      = _restore_boundary_value(d["boundary"]),
        dim           = Int(d["dim"]),
        err_method    = Symbol(d["err_method"]),
        nerr_terms    = Int(d["nerr_terms"]),
        fit_terms     = Int(d["fit_terms"]),
        ff_shift      = Int(d["ff_shift"]),
        use_error_jet = Bool(d["use_error_jet"]),
        use_cuda      = get(d, "use_cuda", false),
        real_type     = get(d, "real_type", string(T)),
    )

    nsamples_restored = if haskey(d, "nsamples") && !isempty(d["nsamples"])
        Int.(collect(d["nsamples"]))
    else
        infer_nsamples(base_res)
    end

    return (
        a             = base_res.a,
        b             = base_res.b,
        nsamples      = nsamples_restored,
        h             = base_res.h,
        tuple_h       = base_res.tuple_h,
        avg           = base_res.avg,
        err           = base_res.err,
        rule          = base_res.rule,
        boundary      = base_res.boundary,
        dim           = base_res.dim,
        err_method    = base_res.err_method,
        nerr_terms    = base_res.nerr_terms,
        fit_terms     = base_res.fit_terms,
        ff_shift      = base_res.ff_shift,
        use_error_jet = base_res.use_error_jet,
        use_cuda      = base_res.use_cuda,
        real_type     = base_res.real_type,
    )
end

# ============================================================
# Public summary API
# ============================================================

"""
    generate_summary_dict(res) -> Dict{String,Any}

Generate a human-readable summary dictionary for [`TOML`](https://toml.io/en/) export.

# Function description
This helper builds a simplified dictionary view of a Maranatha result for
inspection and diagnostics.

The summary includes:

- integration metadata,
- scalarized step sizes,
- original per-axis step information when available,
- quadrature estimates,
- total-like scalar error estimates extracted from each error entry, and
- full per-datapoint error decomposition in serialized dictionary form.

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
- The exported `err_total` field is built through [`_err_entry_total`](@ref),
  allowing both residual-based and refinement-based error entries to contribute
  to the same human-readable summary format.
- For rectangular-domain results, both `h` and `tuple_h` are exported so the
  human-readable summary retains the scalar fitting proxy and the original
  per-axis step structure.
"""
function generate_summary_dict(res)
    return Dict(
        "a"             => _toml_safe(res.a),
        "b"             => _toml_safe(res.b),
        "dim"           => Int(res.dim),
        "rule"          => _toml_safe(_storable_rule_value(res.rule)),
        "boundary"      => _toml_safe(_storable_boundary_value(res.boundary)),
        "err_method"    => String(res.err_method),
        "nerr_terms"    => Int(res.nerr_terms),
        "fit_terms"     => Int(res.fit_terms),
        "ff_shift"      => Int(res.ff_shift),
        "use_error_jet" => Bool(res.use_error_jet),
        "use_cuda"      => getproperty(res, :use_cuda),
        "real_type"     => string(getproperty(res, :real_type)),
        "nsamples"      => hasproperty(res, :nsamples) ? _toml_safe(collect(Int.(res.nsamples))) : _toml_safe(infer_nsamples(res)),
        "h"             => _toml_safe(collect(res.h)),
        "tuple_h"       => hasproperty(res, :tuple_h) ? _toml_safe(res.tuple_h) : _toml_safe(res.h),
        "avg"           => _toml_safe(collect(res.avg)),
        "err_total"     => _toml_safe([_err_entry_total(e) for e in res.err]),
        "err"           => _toml_safe([_err_entry_to_dict(e) for e in res.err]),
    )
end

# ============================================================
# Public save / load API
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
- The optional companion summary is written as a `.toml` file.
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

Return stored subdivision counts when available, otherwise infer them from
step-size information.

# Function description
If `res` already has an `nsamples` field, this helper returns it directly as
`Vector{Int}`.

Otherwise, it reconstructs effective subdivision counts from the relation

```math
N = \\frac{b-a}{h}.
```

It supports both isotropic and rectangular-domain result layouts.

When reconstruction is needed:

* If `tuple_h` is available, each entry is inspected first.
* Tuple- or vector-valued step data are reconstructed axis by axis and checked
  for consistency across axes.
* Scalar step data are checked against the first axis length, and in multi-axis
  cases the same scalar step must also be compatible with every remaining axis.

# Arguments

* `res`: Result object containing domain endpoints and step-size information.

# Keyword arguments

* `atol`: Absolute tolerance used when checking numerical consistency with an
  integer subdivision count.

# Returns

* `Vector{Int}`: Stored or reconstructed subdivision counts.

# Errors

* Throws (via [`JobLoggerTools.error_benji`](@ref)) if a stored step size is not
  numerically consistent with an integer `N`.
* Throws if axis-wise step reconstruction yields inconsistent inferred `N`
  values across axes.
* Throws if a scalar step size is incompatible with a rectangular multi-axis
  domain.

# Notes

* This helper prefers stored `nsamples` when available.
* When reconstruction is needed, it prefers `tuple_h` over `h` because
  `tuple_h` preserves original per-axis step information.
"""
function infer_nsamples(
    res;
    atol = 1e-10,
)
    if hasproperty(res, :nsamples)
        return Int.(collect(res.nsamples))
    end

    T = eltype(res.h)
    atolT = convert(T, atol)
    Ns = Int[]

    a_axes = _to_axis_vector(res.a, res.dim)
    b_axes = _to_axis_vector(res.b, res.dim)
    L_axes = [b_axes[i] - a_axes[i] for i in 1:res.dim]

    tuple_hs = if hasproperty(res, :tuple_h)
        res.tuple_h
    else
        res.h
    end

    for hi in tuple_hs
        if hi isa Tuple || hi isa AbstractVector
            h_axes = _to_axis_vector(hi, res.dim)

            N_candidates = Int[]
            for i in 1:res.dim
                x = L_axes[i] / h_axes[i]
                Ni = round(Int, x)

                isapprox(x, T(Ni); atol = atolT, rtol = zero(T)) || JobLoggerTools.error_benji(
                    "Failed to infer integer N from tuple_h=$(hi) on axis $i"
                )

                push!(N_candidates, Ni)
            end

            all(N -> N == N_candidates[1], N_candidates) || JobLoggerTools.error_benji(
                "Inconsistent inferred N across axes for tuple_h=$(hi): $(N_candidates)"
            )

            push!(Ns, N_candidates[1])
        else
            x = L_axes[1] / hi
            Ni = round(Int, x)

            isapprox(x, T(Ni); atol = atolT, rtol = zero(T)) || JobLoggerTools.error_benji(
                "Failed to infer integer N from h=$hi"
            )

            if res.dim > 1
                for i in 2:res.dim
                    xi = L_axes[i] / hi
                    isapprox(xi, T(Ni); atol = atolT, rtol = zero(T)) || JobLoggerTools.error_benji(
                        "Scalar h=$hi is incompatible with rectangular multi-axis domain."
                    )
                end
            end

            push!(Ns, Ni)
        end
    end

    return Ns
end

# ============================================================
# Public merge / drop API
# ============================================================

"""
    merge_datapoint_results(
        results...;
        sort_by_h=true,
        allow_duplicate_h=false
    ) -> NamedTuple

Merge multiple compatible datapoint result blocks into one result object.

# Function description
This routine concatenates the aligned datapoint arrays

- `nsamples`
- `h`
- `tuple_h`
- `avg`
- `err`

from several compatible result objects, optionally checks for duplicate scalar
step sizes in `h`, and optionally sorts the merged datapoints by descending `h`.

For each input result, `nsamples` are taken from the stored field when present
and otherwise reconstructed via [`infer_nsamples`](@ref).

Global metadata fields are copied from the first input result after shape
compatibility has been verified.

# Arguments
- `results...`: Compatible result objects to merge.

# Keyword arguments
- `sort_by_h`: Whether to sort the merged datapoints by descending `h`.
- `allow_duplicate_h`: Whether duplicate `h` values are allowed.

# Returns
- `NamedTuple`: Merged result object with an explicit `nsamples` field.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two results
  are provided.
- Throws if result metadata is incompatible or if aligned datapoint arrays have
  inconsistent lengths.
- Throws if duplicate `h` values are found and `allow_duplicate_h == false`.
- Propagates reconstruction errors from [`infer_nsamples`](@ref) when an input
  result does not store `nsamples`.

# Notes
- This helper performs metadata-based safety checks, not full semantic identity
  checks on the originating problem.
- For rectangular-domain results, `tuple_h` is merged alongside `h` so the
  original per-axis step information is preserved after concatenation.
- When `sort_by_h == true`, the same permutation is applied to `nsamples`,
  `h`, `tuple_h`, `avg`, and `err`.
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

    T = eltype(result_list[1].h)

    h_all = T[]
    avg_all = T[]
    err_all = eltype(result_list[1].err)[]
    tuple_h_all = Any[]
    nsamples_all = Int[]

    for (i, res) in enumerate(result_list)
        length(res.h) == length(res.avg) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(avg)"
        )
        length(res.h) == length(res.err) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(err)"
        )

        local_tuple_h = hasproperty(res, :tuple_h) ? res.tuple_h : res.h
        length(res.h) == length(local_tuple_h) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(tuple_h)"
        )

        local_nsamples = hasproperty(res, :nsamples) ? Int.(collect(res.nsamples)) : infer_nsamples(res)
        length(res.h) == length(local_nsamples) || JobLoggerTools.error_benji(
            "Length mismatch in result $i: length(h) != length(nsamples)"
        )

        append!(h_all, T.(res.h))
        append!(avg_all, T.(res.avg))
        append!(err_all, res.err)
        append!(tuple_h_all, local_tuple_h)
        append!(nsamples_all, local_nsamples)
    end

    if !allow_duplicate_h
        _assert_no_duplicate_h(h_all)
    end

    if sort_by_h
        p = sortperm(h_all; rev = true)
        h_all = h_all[p]
        avg_all = avg_all[p]
        err_all = err_all[p]
        tuple_h_all = tuple_h_all[p]
        nsamples_all = nsamples_all[p]
    end

    ref = result_list[1]

    return (
        a             = ref.a,
        b             = ref.b,
        nsamples      = nsamples_all,
        h             = h_all,
        tuple_h       = tuple_h_all,
        avg           = avg_all,
        err           = err_all,
        rule          = ref.rule,
        boundary      = ref.boundary,
        dim           = ref.dim,
        err_method    = ref.err_method,
        nerr_terms    = ref.nerr_terms,
        fit_terms     = ref.fit_terms,
        ff_shift      = ref.ff_shift,
        use_error_jet = ref.use_error_jet,
        use_cuda      = getproperty(ref, :use_cuda),
        real_type     = getproperty(ref, :real_type),
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
        name_suffix::String = "merged",
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
- `name_suffix::String`: Filename suffix used when generating a default path.

# Returns
- `String`: Output path of the written merged `.jld2` file.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two input
  files are supplied.
- Propagates loading, merging, path-construction, and saving errors.

# Notes
- The merged [`TOML`](https://toml.io/en/) summary is regenerated from the
  merged in-memory result.
- If `output_path` is not provided, the output filename is generated from the
  merged metadata and inferred subdivision-count list.
"""
function merge_datapoint_result_files(
    paths::AbstractString...;
    output_path::Union{Nothing,AbstractString} = nothing,
    write_summary::Bool = true,
    sort_by_h::Bool = true,
    allow_duplicate_h::Bool = false,
    output_dir::String = ".",
    name_prefix::String = "Maranatha",
    name_suffix::String = "merged",
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
        Ns = hasproperty(merged, :nsamples) ? Int.(collect(merged.nsamples)) : infer_nsamples(merged)
        _default_result_path(
            output_dir,
            name_prefix,
            name_suffix,
            merged.a,
            merged.b,
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
This helper builds a keep-mask from the result's subdivision-count sequence and
returns a filtered copy of the input result with all datapoint-aligned arrays
reduced consistently, including `tuple_h` when present.

If `res` already stores `nsamples`, those values are used directly. Otherwise,
the sequence is reconstructed via [`infer_nsamples`](@ref).

# Arguments
- `res`: Result object to filter.
- `Ns_to_drop`: Collection of subdivision counts to remove.

# Keyword arguments
- `atol`: Absolute tolerance used when reconstructing subdivision counts.

# Returns
- `NamedTuple`: Filtered result object with an explicit `nsamples` field.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if filtering would remove
  all datapoints.
- Propagates reconstruction errors from [`infer_nsamples`](@ref) when
  `nsamples` must be rebuilt from step-size information.

# Notes
- Global metadata fields are copied from the original result, including
  `use_cuda` and `real_type` when present.
- If the result stores `tuple_h`, that per-axis step information is filtered in
  sync with `h`, `avg`, and `err`; otherwise `h` is reused as the fallback
  source for `tuple_h`.
"""
function drop_nsamples_from_result(
    res,
    Ns_to_drop;
    atol::Float64 = 1e-10,
)
    Ns_all = hasproperty(res, :nsamples) ? Int.(collect(res.nsamples)) : infer_nsamples(res; atol=atol)

    drop_set = Set(Int.(Ns_to_drop))

    keep_mask = [!(N in drop_set) for N in Ns_all]

    any(keep_mask) || JobLoggerTools.error_benji(
        "drop_nsamples_from_result would remove all datapoints."
    )

    nsamples_new = Ns_all[keep_mask]
    h_new       = res.h[keep_mask]
    tuple_h_new = hasproperty(res, :tuple_h) ? res.tuple_h[keep_mask] : res.h[keep_mask]
    avg_new     = res.avg[keep_mask]
    err_new     = res.err[keep_mask]

    return (
        a             = res.a,
        b             = res.b,
        nsamples      = nsamples_new,
        h             = h_new,
        tuple_h       = tuple_h_new,
        avg           = avg_new,
        err           = err_new,
        rule          = res.rule,
        boundary      = res.boundary,
        dim           = res.dim,
        err_method    = res.err_method,
        nerr_terms    = res.nerr_terms,
        fit_terms     = res.fit_terms,
        ff_shift      = res.ff_shift,
        use_error_jet = res.use_error_jet,
        use_cuda      = getproperty(res, :use_cuda),
        real_type     = getproperty(res, :real_type),
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
        name_prefix::String = "Maranatha",
        name_suffix::String = "filtered",
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
- `name_suffix::String`: Filename suffix used when generating a default path.

# Returns
- `String`: Output path of the written filtered `.jld2` file.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if filtering would remove
  all datapoints.
- Propagates loading, filtering, path-construction, and saving errors.

# Notes
- The filtered result preserves the stored global metadata from the original file.
- If `output_path` is not provided, the output filename is generated from the
  filtered subdivision-count list.
"""
function drop_nsamples_from_file(
    input_path::AbstractString,
    Ns_to_drop;
    output_path::Union{Nothing,AbstractString} = nothing,
    write_summary::Bool = true,
    atol::Float64 = 1e-10,
    output_dir::String = ".",
    name_prefix::String = "Maranatha",
    name_suffix::String = "filtered",
)
    result = load_datapoint_results(input_path)

    filtered = drop_nsamples_from_result(
        result,
        Ns_to_drop;
        atol = atol,
    )

    final_output_path = if output_path === nothing
        Ns = hasproperty(filtered, :nsamples) ? Int.(collect(filtered.nsamples)) : infer_nsamples(filtered; atol=atol)
        _default_result_path(
            output_dir,
            name_prefix,
            name_suffix,
            filtered.a,
            filtered.b,
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
