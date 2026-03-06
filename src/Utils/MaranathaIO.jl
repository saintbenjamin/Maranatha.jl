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

using ..TOML
using ..JLD2

using ..Utils.JobLoggerTools

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
            center      = Float64(e["center"]),
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

    jldsave(path; datapoint_results=d)

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

end  # module IO