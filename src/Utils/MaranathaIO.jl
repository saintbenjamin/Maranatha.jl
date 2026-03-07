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

Convert a Maranatha result `NamedTuple` into a plain `Dict`
structure suitable for serialization.

# Function description

The result produced by [`Maranatha.Runner.run_Maranatha`](@ref) 
contains nested `NamedTuple` structures, 
including a vector of per-sample error descriptions.
This helper converts that structure into a dictionary composed only
of standard serializable containers:

- `Dict`
- `Vector`
- `Int`
- `Float64`
- `String`

This representation is intended for use with external storage formats
such as **JLD2** and **TOML**, which do not reliably preserve
Julia-specific structures like `NamedTuple` with symbolic fields.

# Structure

The conversion preserves the full information contained in the result:

- global integration metadata (`a`, `b`, `rule`, `boundary`, etc.)
- step sizes `h`
- quadrature estimates `avg`
- detailed per-sample error decomposition

Each element of `res.err` (originally a `NamedTuple`) is converted into
a `Dict` containing

```
ks          residual moment indices
coeffs      residual coefficients
derivatives evaluated derivatives of the integrand
terms       individual error contributions
total       summed truncation estimate
center      expansion center
h           step size used for the estimate
```

The resulting dictionary is therefore fully self-contained and can
later be reconstructed into the original structure via
[`dict_to_namedtuple`](@ref).
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

Reconstruct a quadrature result `NamedTuple` from a serialized dictionary.

# Function description

This routine performs the inverse operation of
[`namedtuple_to_dict`](@ref).

It restores the original Julia data structure used internally by
[`Maranatha.Runner.run_Maranatha`](@ref), converting the serialized dictionary representation
back into a structured `NamedTuple`.

# Reconstruction steps

The following conversions are applied:

- numeric vectors (`h`, `avg`) are restored as `Vector{Float64}`
- rule and boundary identifiers are restored as `Symbol`
- each entry in `d["err"]` is reconstructed as a `NamedTuple`
- the stored error-expansion center `center` is restored as either
  a scalar `Float64` or a tuple of `Float64`, depending on whether
  the serialized representation corresponds to a one-dimensional or
  multi-dimensional expansion center

The resulting structure matches the layout expected by downstream
analysis routines such as 
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
and can therefore be passed directly into fitting or diagnostic
pipelines.

# Notes

The dictionary `d` is assumed to originate from
[`namedtuple_to_dict`](@ref).  No validation of external schema
compatibility is performed.
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

Generate a human-readable summary dictionary for TOML export.

# Function description

This helper constructs a simplified representation of a Maranatha.jl
integration result intended for **inspection and diagnostics**, rather
than exact reconstruction.

The produced dictionary contains

- integration metadata (`rule`, `boundary`, `dim`, etc.)
- step sizes `h`
- quadrature estimates `avg`
- total error estimates for each sample
- full error decomposition entries

This representation is later written to a `*.toml` file via
`TOML.print`.

# Purpose

While JLD2 files preserve the full binary representation of the result,
the TOML summary serves as a **human-readable companion file** that
allows quick inspection of

- convergence behaviour
- estimated error scales
- integration configuration

without requiring Julia to load the binary data.

# Relationship to serialization

Unlike [`namedtuple_to_dict`](@ref), this function is not intended for
lossless reconstruction of the original result structure.
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

Save a Maranatha integration result to disk.

# Function description

This routine serializes the result produced by 
[`Maranatha.Runner.run_Maranatha`](@ref) into a
**JLD2 binary file**.

Internally the result `NamedTuple` is first converted into a plain
dictionary representation using [`namedtuple_to_dict`](@ref), ensuring
that the stored structure contains only serialization-safe data types.

# File outputs

Two files may be written:

• **JLD2 file**
```
path
```
Contains the full result under the dataset key
```
"datapoint_results"
```

• **TOML summary (optional)**

If `write_summary=true`, a companion file
```
path → path with extension `.toml`
```
is written containing a human-readable summary generated by
[`generate_summary_dict`](@ref).

# Arguments

`path`
: Destination `.jld2` file.

`res`
: Result `NamedTuple` produced by `run_Maranatha`.

`write_summary`
: Whether to additionally write a TOML summary file.

# Returns

The path to the written `.jld2` file.

# Notes

The function enforces that `path` ends with `.jld2`, since the stored
data structure is intended for JLD2 serialization.
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

Load a previously saved Maranatha integration result.

# Function description

This routine reads a `.jld2` file produced by
[`save_datapoint_results`](@ref) and reconstructs the original
quadrature result structure.

The stored dictionary representation is converted back into a Julia
`NamedTuple` via [`dict_to_namedtuple`](@ref), restoring

- quadrature estimates
- step-size sequence
- detailed error decomposition
- integration metadata

# Arguments

`path`
: Path to a `.jld2` file containing a stored result.

# Returns

A `NamedTuple` compatible with downstream analysis routines such as
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)

# Notes

The file must contain the dataset key
```
"datapoint_results"
```
which is the format produced by
[`save_datapoint_results`](@ref).
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

Infer the subdivision counts ``N`` from the stored step sizes ``h``
contained in a Maranatha result object.

# Function description

This helper reconstructs the effective sample counts used in
[`Maranatha.Runner.run_Maranatha`](@ref) from the relation
```math
N = \\frac{b-a}{h} \\,.
```

Since the result object stores step sizes ``h`` rather than the original
input vector `nsamples`, this routine provides a convenient way to
recover the corresponding integer subdivision counts for inspection,
diagnostics, or file naming.

# Reconstruction rule

For each stored step size `h_i`, the routine computes
```math
N_i = \\mathrm{round}\\!\\left(\\frac{b-a}{h_i}\\right)
```
and verifies that the floating-point value is sufficiently close to an
integer within the tolerance specified by `atol`.

# Arguments

`res`
: Result object produced by [`Maranatha.Runner.run_Maranatha`](@ref)
or reconstructed by [`load_datapoint_results`](@ref).

`atol`
: Absolute tolerance used when checking whether ``\\dfrac{b-a}{h}``
is numerically consistent with an integer.

# Returns

A vector of inferred integer subdivision counts.

# Notes

This routine assumes that each stored ``h`` truly arose from the standard
Maranatha convention
```math
h = \\frac{b-a}{N} \\,.
```

If the stored step sizes are inconsistent with that relation, an error
is raised.
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

Construct a filename-friendly suffix that explicitly lists the subdivision
counts ``N`` present in a result.

# Function description

This internal helper converts a collection of subdivision counts into the
compact string form used by Maranatha result filenames.

For example,

```julia
[2, 3, 4, 5, 6, 7]
```

is converted into

```julia
"N_2_3_4_5_6_7"
```

This makes saved result files reflect the actual datapoints present in the
dataset, rather than only the first and last sample counts.

# Arguments

`Ns`
: Collection of subdivision counts to encode into a filename suffix.

# Returns

A string of the form

```julia
N_\$(N1)_\$(N2)_\$(N3)_...
```

suitable for inclusion in `.jld2` and `.toml` result filenames.

# Notes

This helper is intended for internal path construction utilities such as
[`_default_result_path`](@ref).
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

Construct the default output path for a Maranatha result file.

# Function description

This internal helper builds a standard `.jld2` filename using

* a target directory
* a user-facing name prefix
* the quadrature rule label
* the boundary label
* the explicit list of subdivision counts present in the result

The subdivision-count portion of the filename is generated via
[`_build_nsamples_suffix`](@ref), so that the saved path records the
actual datapoints contained in the file.

For example, a result with

* `name_prefix = "merged"`
* `rule = :gauss_p4`
* `boundary = :LU_EXEX`
* `Ns = [2,3,4,5,6,7]`

produces a path ending in

```julia
result_merged_gauss_p4_LU_EXEX_N_2_3_4_5_6_7.jld2
```

# Arguments

`save_dir`
: Directory in which the result file should be placed.

`name_prefix`
: User-facing filename prefix describing the dataset.

`rule`
: Quadrature rule label.

`boundary`
: Boundary-condition label.

`Ns`
: Collection of subdivision counts present in the result.

# Returns

A full path to the default `.jld2` output file.

# Notes

This helper only constructs the path string.  It does not create any files
or directories by itself.
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

Check that multiple Maranatha result objects are mutually compatible
for datapoint-level merging.

# Function description

This internal helper verifies that a collection of result blocks shares
the same global integration metadata before their datapoint arrays are
combined by [`merge_datapoint_results`](@ref).

The merge operation is only valid when all input results correspond to
the same computational setup, differing only in the specific set of
step sizes ``h`` (or equivalently, subdivision counts ``N``) that were
evaluated.

# Checked fields

The following fields must agree across all input results:

- `a`
- `b`
- `dim`
- `rule`
- `boundary`
- `err_method`
- `nerr_terms`
- `fit_terms`
- `ff_shift`
- `use_threads`

If any mismatch is detected, an error is raised with a message
indicating which field failed and at which result index.

# Arguments

`results`
: A collection of result objects produced by
  [`Maranatha.Runner.run_Maranatha`](@ref) or loaded via
  [`load_datapoint_results`](@ref).

# Returns

`nothing`

# Notes

This helper checks structural compatibility of the stored metadata,
but it does not and cannot prove that the original integrand object
was identical across runs.  It therefore serves as a metadata-based
merge safety check, not a full semantic identity check.
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

Check that a collection of step sizes ``h`` contains no duplicate entries
up to a specified absolute tolerance.

# Function description

This internal helper is used during datapoint-result merging to prevent
multiple result blocks from contributing the same effective sample
location more than once.

Since Maranatha stores convergence data as aligned arrays of

- step sizes ``h``
- quadrature estimates `avg`
- error descriptors `err`

duplicated `h` values would generally indicate overlapping runs, such as
two files both containing results for the same subdivision count ``N``.

# Detection rule

The input step sizes are first sorted, after which adjacent values are
compared using `isapprox(...; atol=atol, rtol=0.0)`.  If any pair is
found to be numerically identical within tolerance, an error is raised.

# Arguments

`hs`
: Collection of step sizes to be checked.

`atol`
: Absolute tolerance used when comparing nearby sorted values.

# Returns

`nothing`

# Notes

This helper is intentionally conservative.  Its purpose is not to merge
nearby points, but to reject potentially ambiguous duplicate entries
before constructing a combined result object.
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

Merge multiple Maranatha datapoint result blocks into a single combined
result object.

# Function description

This routine combines several compatible result objects, each typically
produced by separate calls to [`Maranatha.Runner.run_Maranatha`](@ref),
into one unified result with concatenated datapoint arrays.

A common use case is to evaluate a long sequence of subdivision counts
in several batches, for example

```julia
[2,3], [4,5], [6,7,8]
```

and then merge the separately saved results into one object equivalent
to a single run over

```julia
[2,3,4,5,6,7,8] .
```

# Merge rule

For each input result, the following aligned arrays are concatenated:

* ``h``
* `avg`
* `err`

All global metadata fields are copied from the first result after
compatibility is verified via [`_assert_same_result_shape`](@ref).

# Keyword arguments

`sort_by_h`
: If `true`, the merged datapoints are sorted by descending `h`
(equivalently, from smaller `N` to larger `N` under the standard
relation `h=(b-a)/N`).

`allow_duplicate_h`
: If `false`, duplicate step sizes are rejected via
[`_assert_no_duplicate_h`](@ref).  If `true`, overlapping `h`
values are allowed and preserved as-is.

# Returns

A merged result `NamedTuple` having the same structure as the output of
[`Maranatha.Runner.run_Maranatha`](@ref).

# Notes

This routine performs metadata-based compatibility checks, but it does
not verify the identity of the original integrand object itself.
Therefore, it should only be used when the caller knows that all input
results came from the same physical or numerical problem setup.
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

Load multiple saved Maranatha result files, merge them, and write the
combined result back to disk.

# Function description

This routine is a file-based convenience wrapper around
[`merge_datapoint_results`](@ref).

Each input path is first loaded via [`load_datapoint_results`](@ref),
after which the resulting objects are merged in memory.  The combined
result is then written to `output_path` using
[`save_datapoint_results`](@ref).

# Typical use case

This helper is intended for cases where a larger convergence run was
split across multiple partial jobs, for example due to time limits,
batch execution, or interrupted sessions, and the partial `.jld2`
results later need to be reassembled into one complete dataset.

# Keyword arguments

`output_path`
: Destination path of the merged `.jld2` file.

`write_summary`
: Whether to additionally write the companion TOML summary file.

`sort_by_h`
: Whether to sort the merged datapoints by descending `h`.

`allow_duplicate_h`
: Whether duplicate `h` values are permitted during merging.

# Returns

The output path of the written merged `.jld2` file.

# Notes

This routine treats the saved `.jld2` files as the authoritative source
and regenerates the merged TOML summary from the merged in-memory result,
rather than attempting any direct TOML-to-TOML merge.
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

Remove selected subdivision counts ``N`` from a Maranatha result object.

# Function description

This helper constructs a filtered copy of an existing result produced by
[`Maranatha.Runner.run_Maranatha`](@ref), removing datapoints corresponding
to specified subdivision counts ``N``.

The original result stores only the step sizes ``h`` rather than the input
vector `nsamples`.  The corresponding subdivision counts are therefore
reconstructed internally using
```math
N = \\frac{b-a}{h}
```
via [`infer_nsamples`](@ref).  Any datapoint whose inferred `N` belongs
to `Ns_to_drop` is excluded from the returned result.

All datapoint-aligned arrays

* `h`
* `avg`
* `err`

are filtered consistently using the same mask.

# Arguments

`res`
: Result object produced by [`Maranatha.Runner.run_Maranatha`](@ref)
or reconstructed via [`load_datapoint_results`](@ref).

`Ns_to_drop`
: Collection of subdivision counts `N` that should be removed
from the result.

# Keyword arguments

`atol`
: Absolute tolerance used when reconstructing subdivision counts
from the stored step sizes `h`.

# Returns

A new result `NamedTuple` with the same metadata as the input result
but with the specified datapoints removed.

# Errors

* Throws an error if the filtering would remove **all datapoints**.
* Propagates errors from [`infer_nsamples`](@ref) if the stored step
  sizes are inconsistent with the relation `h=(b-a)/N`.

# Notes

This routine modifies only the datapoint arrays.  All global metadata
fields (`rule`, `boundary`, `dim`, etc.) are copied unchanged from the
original result.
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
    function drop_nsamples_from_file(
        input_path::AbstractString,
        Ns_to_drop;
        output_path::Union{Nothing,AbstractString} = nothing,
        write_summary::Bool = true,
        atol::Float64 = 1e-10,
        output_dir::String = ".",
        name_prefix::String = "dropped",
    ) -> String

Remove selected subdivision counts `N` from a saved Maranatha result file.

# Function description

This routine is a file-level convenience wrapper around
[`drop_nsamples_from_result`](@ref).

It loads an existing `.jld2` result file produced by
[`save_datapoint_results`](@ref), removes datapoints corresponding to the
specified subdivision counts `N`, and writes the filtered result back
to disk.

# Workflow

The procedure consists of three steps:

1. load the result via [`load_datapoint_results`](@ref)
2. filter datapoints using [`drop_nsamples_from_result`](@ref)
3. write the filtered result via [`save_datapoint_results`](@ref)

# Arguments

`input_path`
: Path to the source `.jld2` result file.

`Ns_to_drop`
: Collection of subdivision counts `N` to be removed.

# Keyword arguments

`output_path`
: Destination `.jld2` file containing the filtered result.

`write_summary`
: Whether to write the companion TOML summary file.

`atol`
: Absolute tolerance used when reconstructing subdivision counts
from stored step sizes.

# Returns

The output path of the written `.jld2` file.

# Errors

* Throws an error if the filtering would remove **all datapoints**.
* Propagates errors from the underlying load, filtering, or save
  routines.

# Notes

The resulting file preserves all global metadata fields from the
original result while containing only the selected subset of datapoints.
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