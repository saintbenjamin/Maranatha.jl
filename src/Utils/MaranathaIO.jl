# ============================================================================
# src/Utils/MaranathaIO.jl
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
# 1. Serialization helpers
# ============================================================

# ------------------------------------------------------------
# Geometry / TOML helper utilities
# ------------------------------------------------------------

"""
    _is_scalar_domain_value(x) -> Bool

Return `true` if `x` represents a scalar domain value.

# Function description

This predicate distinguishes scalar endpoints from multi-axis domain
representations. A value is considered scalar if it is neither a `Tuple`
nor an `AbstractVector`.

It is typically used when determining whether a problem domain is
one-dimensional (scalar) or multi-dimensional (tuple/vector-based).

# Arguments

- `x`: Domain endpoint candidate.

# Returns

- `Bool`: `true` if `x` is scalar-like, `false` if it represents a
  multi-component domain.
"""
@inline _is_scalar_domain_value(x) = !(x isa Tuple) && !(x isa AbstractVector)

"""
    _storable_domain_value(x)

Convert a domain value into a container suitable for serialization.

# Function description

This helper normalizes domain endpoints for storage (e.g., in TOML or JSON)
by converting tuple- or vector-like objects into concrete vectors while
leaving scalar values unchanged.

The goal is to ensure that serialized representations use standard,
portable container types.

# Arguments

- `x`: Domain value (scalar, tuple, or vector-like).

# Returns

- A storable representation:

  - `Vector` if `x` is a tuple or vector-like object,
  - the original value if `x` is scalar.
"""
@inline function _storable_domain_value(x)
    if x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return x
    end
end

"""
    _storable_boundary_value(x)

Convert a boundary specification into a serialization-friendly representation.

# Function description
Scalar boundary symbols are converted to strings. Tuple/vector axis-wise
boundary specifications are converted to vectors of strings. Other inputs are
returned unchanged.

# Arguments
- `x`: Boundary specification to normalize for storage.

# Returns
- A storage-friendly boundary representation composed of strings and standard
  containers.

# Errors
- No explicit validation is performed.
"""
@inline function _storable_boundary_value(x)
    if x isa Symbol
        return String(x)
    elseif x isa Tuple
        return String.(collect(x))
    elseif x isa AbstractVector
        return String.(collect(x))
    else
        return x
    end
end

"""
    _storable_rule_value(x)

Convert a quadrature-rule specification into a serialization-friendly
representation.

# Function description
Scalar rule symbols are converted to strings. Tuple/vector axis-wise rule
specifications are converted to vectors of strings. Other inputs are returned
unchanged.

# Arguments
- `x`: Rule specification to normalize for storage.

# Returns
- A storage-friendly rule representation composed of strings and standard
  containers.

# Errors
- No explicit validation is performed.
"""
@inline function _storable_rule_value(x)
    if x isa Symbol
        return String(x)
    elseif x isa Tuple
        return String.(collect(x))
    elseif x isa AbstractVector
        return String.(collect(x))
    else
        return x
    end
end

"""
    _restore_rule_value(x)

Reconstruct a quadrature-rule specification from serialized storage form.

# Function description
String input is converted back to a scalar `Symbol`. Vector input is converted
to a tuple of `Symbol` values, restoring the axis-wise rule form used by the
runtime pipeline.

# Arguments
- `x`: Serialized rule representation.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Conversion errors propagate if the stored representation is malformed.
"""
@inline function _restore_rule_value(x)
    if x isa AbstractString
        return Symbol(x)
    elseif x isa AbstractVector
        return Tuple(Symbol.(x))
    else
        return Symbol(x)
    end
end

"""
    _restore_domain_value(x, T)

Reconstruct a domain value from a serialized representation.

# Function description

This routine reverses the transformation applied by
[`_storable_domain_value`](@ref), converting stored vector data back
into the domain representation expected by the computational code.

If the input is vector-like, the elements are converted to type `T`
and returned as a tuple. Scalar inputs are converted directly to `T`.

# Arguments

- `x`: Stored domain value.
- `T`: Target numeric type.

# Returns

- A domain value of type `T`:

  - `Tuple{T,...}` if `x` is vector-like,
  - scalar `T` otherwise.
"""
@inline function _restore_domain_value(x, T)
    if x isa AbstractVector
        return Tuple(convert.(T, x))
    else
        return convert(T, x)
    end
end

"""
    _restore_boundary_value(x)

Reconstruct a boundary specification from serialized storage form.

# Function description
String input is converted back to a scalar boundary `Symbol`. Vector input is
converted to a tuple of `Symbol` values, restoring the axis-wise boundary form
used by the runtime pipeline.

# Arguments
- `x`: Serialized boundary representation.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Conversion errors propagate if the stored representation is malformed.
"""
@inline function _restore_boundary_value(x)
    if x isa AbstractString
        return Symbol(x)
    elseif x isa AbstractVector
        return Tuple(Symbol.(x))
    else
        return Symbol(x)
    end
end

"""
    _filename_spec_is_multi(x) -> Bool

Return `true` if `x` participates as an axis-wise filename specification.

# Function description
Tuple and vector values are classified as axis-wise metadata, while all other
values are treated as shared scalar metadata for filename construction.

# Arguments
- `x`: Candidate filename-metadata value.

# Returns
- `Bool`: `true` for tuple/vector input, `false` otherwise.

# Errors
- No explicit validation is performed.
"""
@inline _filename_spec_is_multi(x) = x isa Tuple || x isa AbstractVector

"""
    _filename_spec_dim(a, b, rule, boundary) -> Int

Infer the effective axis count used for result-filename construction.

# Function description
This helper inspects domain bounds together with rule and boundary metadata and
returns the unique common axis count implied by any axis-wise inputs. If all
inputs are scalar-like, the returned dimension is `1`.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `Int`: Effective dimension used when expanding filename tokens.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `a` and `b` mix scalar and
  collection styles or if axis-wise inputs imply inconsistent dimensions.
"""
@inline function _filename_spec_dim(a, b, rule, boundary)::Int
    a_multi = _filename_spec_is_multi(a)
    b_multi = _filename_spec_is_multi(b)

    a_multi == b_multi || JobLoggerTools.error_benji(
        "Filename-spec mismatch: `a` and `b` must both be scalar or both be tuple/vector-like."
    )

    dims = Int[]

    if a_multi
        push!(dims, length(a))
        push!(dims, length(b))
    end
    _filename_spec_is_multi(rule)     && push!(dims, length(rule))
    _filename_spec_is_multi(boundary) && push!(dims, length(boundary))

    isempty(dims) && return 1

    dim = first(dims)
    all(==(dim), dims) || JobLoggerTools.error_benji(
        "Filename-spec mismatch: inconsistent axis counts across domain/rule/boundary."
    )

    return dim
end

"""
    _filename_spec_at(x, d::Int, dim::Int)

Resolve one filename-metadata value on axis `d`.

# Function description
Scalar inputs are treated as shared values and returned unchanged. Tuple/vector
inputs are validated against `dim` and indexed at axis `d`.

# Arguments
- `x`: Scalar or axis-wise filename-metadata specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise inputs.

# Returns
- The scalar shared value or the axis-local entry `x[d]`.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if an axis-wise input has
  length different from `dim`.
"""
@inline function _filename_spec_at(x, d::Int, dim::Int)
    if _filename_spec_is_multi(x)
        length(x) == dim || JobLoggerTools.error_benji(
            "Filename-spec mismatch: spec length must equal dim."
        )
        return x[d]
    end
    return x
end

"""
    _rule_boundary_filename_token(a, b, rule, boundary) -> String

Construct the compact rule/boundary token used in saved-result filenames.

# Function description
If all inputs are scalar-like, this helper returns `"<rule>_<boundary>"`. If
any input is axis-wise, it expands the result into an axis-tagged token of the
form `"1_<rule1>_<boundary1>_2_<rule2>_<boundary2>_..."`.

# Arguments
- `a`, `b`: Domain-bound specifications used only to infer scalar vs axis-wise
  filename layout.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `String`: Filename-friendly rule/boundary token.

# Errors
- Propagates dimensional-consistency errors from
  [`_filename_spec_dim`](@ref) and [`_filename_spec_at`](@ref).

# Notes
- Domain values themselves are not embedded in the returned token.
"""
@inline function _rule_boundary_filename_token(a, b, rule, boundary)::String
    dim = _filename_spec_dim(a, b, rule, boundary)

    if dim == 1
        return "$(string(rule))_$(string(boundary))"
    end

    parts = String[]
    for d in 1:dim
        push!(parts, string(d))
        push!(parts, string(_filename_spec_at(rule, d, dim)))
        push!(parts, string(_filename_spec_at(boundary, d, dim)))
    end

    return join(parts, "_")
end

"""
    _to_axis_vector(x, dim::Int) -> Vector

Convert a domain specification into a concrete vector form.

# Function description

This helper converts scalar and collection-valued domain data into the vector
representation used by internal geometry and `nsamples` reconstruction code.

The current behavior is:

- `Tuple` input: returns `[x[i] for i in 1:dim]`, so the first `dim` entries
  are copied and a bounds error is raised if the tuple is shorter than `dim`.
- `AbstractVector` input: returns `collect(x)` without enforcing that the
  resulting length matches `dim`.
- Scalar input: returns `fill(x, dim)`.

# Arguments

- `x`: Domain value (scalar, tuple, or vector-like).
- `dim`: Target dimensionality used for scalar expansion and tuple indexing.

# Returns

- `Vector`: Collected axis values. Scalar input yields a length-`dim` vector;
  tuple and vector input follow the rules above.

# Errors

- May throw a bounds error if `x isa Tuple` and `length(x) < dim`.
"""
@inline function _to_axis_vector(x, dim::Int)
    if x isa Tuple
        return [x[i] for i in 1:dim]
    elseif x isa AbstractVector
        return collect(x)
    else
        return fill(x, dim)
    end
end

"""
    _toml_safe(x)

Convert an arbitrary object into a TOML-compatible representation.

# Function description

This recursive helper transforms complex Julia objects into values that
can be safely serialized in TOML format.

The conversion rules aim to preserve meaning while ensuring compatibility
with TOML's restricted data model.

Supported conversions include:

- Primitive TOML types (`Bool`, integers, strings) → unchanged
- Floating-point values → preserved if standard (`Float32`, `Float64`),
  otherwise converted to string
- Symbols → converted to strings
- Tuples and vectors → converted elementwise to arrays
- Dictionaries and named tuples → converted to `Dict{String,Any}`
  with recursively processed values
- Other types → converted to string representations

# Arguments

- `x`: Arbitrary Julia object.

# Returns

- A TOML-safe value composed only of supported scalar types,
  arrays, and string-keyed dictionaries.

# Notes

- This function prioritizes robustness over round-trip fidelity.
- Non-standard numeric types may be stringified to avoid loss of
  information during serialization.
"""
@inline function _toml_safe(x)
    if x isa Bool || x isa Integer || x isa AbstractString
        return x
    elseif x isa AbstractFloat
        return x isa Float32 || x isa Float64 ? x : string(x)
    elseif x isa Symbol
        return String(x)
    elseif x isa Tuple
        return [_toml_safe(v) for v in x]
    elseif x isa AbstractVector
        return [_toml_safe(v) for v in x]
    elseif x isa Dict
        out = Dict{String,Any}()
        for (k, v) in x
            out[string(k)] = _toml_safe(v)
        end
        return out
    elseif x isa NamedTuple
        out = Dict{String,Any}()
        for k in keys(x)
            out[string(k)] = _toml_safe(getproperty(x, k))
        end
        return out
    else
        return string(x)
    end
end

"""
    _err_entry_to_dict(e) -> Dict{String,Any}

Convert a single internal error-entry object into a serialization-friendly dictionary.

# Function description
This helper normalizes one element of `res.err` into a plain dictionary composed
of standard scalar and container types suitable for storage in `JLD2`, `TOML`,
or related external formats.

This conversion supports both scalar-step and axis-wise-step error entries.
Fields such as `:center`, `:h`, `:h_coarse`, and `:h_fine` are preserved through
`_storable_domain_value`, so rectangular-domain metadata can be serialized
without losing per-axis structure.

It currently supports two error-entry layouts:

- residual-based error objects exposing fields such as `:ks`, `:coeffs`,
  `:derivatives`, `:terms`, and `:total`, and
- refinement-based error objects exposing fields such as `:method`,
  `:N_coarse`, `:N_fine`, and `:estimate`.

# Arguments
- `e`:
  One internal error-entry object, typically an element of `res.err`.

# Returns
- `Dict{String,Any}`:
  Serialization-friendly dictionary representation of the error entry.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `e` does not match any supported
  error-entry structure.

# Notes
- Residual-based entries are currently tagged with `"err_format" => "derivative"`.
- Refinement-based entries are currently tagged with `"err_format" => "refinement"`.
- Residual-based entries are serialized in the legacy flat layout. Any
  axis-wise decomposition stored in a `per_axis` field is not currently
  preserved by this helper.
"""
function _err_entry_to_dict(e)
    if hasproperty(e, :ks)
        return Dict(
            "err_format"   => "derivative",
            "ks"           => collect(Int.(e.ks)),
            "coeffs"       => collect(e.coeffs),
            "derivatives"  => collect(e.derivatives),
            "terms"        => collect(e.terms),
            "total"        => e.total,
            "center"       => _storable_domain_value(e.center),
            "h"            => _storable_domain_value(e.h),
        )
    elseif hasproperty(e, :estimate)
        return Dict(
            "err_format"   => "refinement",
            "method"       => String(e.method),
            "rule"         => _storable_rule_value(e.rule),
            "boundary"     => _storable_boundary_value(e.boundary),
            "N_coarse"     => Int(e.N_coarse),
            "N_fine"       => Int(e.N_fine),
            "dim"          => Int(e.dim),
            "h_coarse"     => _storable_domain_value(e.h_coarse),
            "h_fine"       => _storable_domain_value(e.h_fine),
            "q_coarse"     => e.q_coarse,
            "q_fine"       => e.q_fine,
            "estimate"     => e.estimate,
            "signed_diff"  => e.signed_diff,
            "reference"    => e.reference,
        )
    else
        JobLoggerTools.error_benji(
            "Unsupported error entry format during serialization. " *
            "Expected derivative or refinement structure."
        )
    end
end

"""
    _dict_to_err_entry(e)

Reconstruct a single internal error-entry object from a serialized dictionary.

# Function description
This helper performs the inverse conversion of [`_err_entry_to_dict`](@ref).
It reads the stored `"err_format"` tag and rebuilds the corresponding internal
error-entry structure expected by downstream fitting, plotting, and reporting
code.

Currently supported serialized formats are:

- `"derivative"` for derivative-based error entries, and
- `"refinement"` for refinement-based error entries.

# Arguments
- `e`:
  Serialized dictionary representation of one error entry.

# Returns
- A reconstructed internal error-entry `NamedTuple`.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if the stored `"err_format"` value
  is unsupported.

# Notes
- The reconstructed structure is intended to match the field layout expected by
  downstream Maranatha workflows.
- Scalar and axis-wise geometric fields are restored through
  [`_restore_domain_value`](@ref), so rectangular-domain error metadata is
  reconstructed as tuples when appropriate.
- Residual-based entries are reconstructed in the legacy flat layout and do
  not currently restore a `per_axis` decomposition.
"""
function _dict_to_err_entry(e)
    fmt = get(e, "err_format", "refinement")

    if fmt == "derivative"
        coeffs = collect(e["coeffs"])
        T = isempty(coeffs) ? Float64 : eltype(coeffs)

        return (
            ks          = Vector{Int}(e["ks"]),
            coeffs      = Vector{T}(coeffs),
            derivatives = Vector{T}(e["derivatives"]),
            terms       = Vector{T}(e["terms"]),
            total       = convert(T, e["total"]),
            center      = _restore_domain_value(e["center"], T),
            h           = _restore_domain_value(e["h"], T),
        )
    elseif fmt == "refinement"
        T = typeof(e["estimate"])
        return (
            method      = Symbol(e["method"]),
            rule        = _restore_rule_value(e["rule"]),
            boundary    = _restore_boundary_value(e["boundary"]),
            N_coarse    = Int(e["N_coarse"]),
            N_fine      = Int(e["N_fine"]),
            dim         = Int(e["dim"]),
            h_coarse    = _restore_domain_value(e["h_coarse"], T),
            h_fine      = _restore_domain_value(e["h_fine"], T),
            q_coarse    = convert(T, e["q_coarse"]),
            q_fine      = convert(T, e["q_fine"]),
            estimate    = convert(T, e["estimate"]),
            signed_diff = convert(T, e["signed_diff"]),
            reference   = convert(T, e["reference"]),
        )
    else
        JobLoggerTools.error_benji(
            "Unsupported err_format during deserialization: err_format=$(fmt)"
        )
    end
end

"""
    _err_entry_total(e) -> Real

Extract the stored scalar error quantity from an internal error-entry object.

# Function description
This helper provides a unified scalar accessor across the currently supported
error-entry layouts.

It supports:

- derivative-style error entries exposing a `:total` field, and
- refinement-style error entries exposing an `:estimate` field.

The stored scalar value is returned as-is; this helper does not convert the
result to `Float64` and does not apply `abs(...)`.

# Arguments
- `e`:
  One internal error-entry object.

# Returns
- `Real`:
  Stored scalar error quantity associated with the entry.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `e` does not expose either
  `:total` or `:estimate`.

# Notes
- This helper is mainly used for summary export and human-readable diagnostics.
"""
function _err_entry_total(e)
    if hasproperty(e, :total)
        return e.total
    elseif hasproperty(e, :estimate)
        return e.estimate
    else
        JobLoggerTools.error_benji(
            "Unsupported error entry format while extracting total-like quantity."
        )
    end
end

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
# 2. TOML summary helpers
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
        res.use_error_jet == ref.use_error_jet || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: use_error_jet differs ($(res.use_error_jet) != $(ref.use_error_jet))"
        )
        getproperty(res, :use_cuda) == getproperty(ref, :use_cuda) || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: use_cuda differs ($(getproperty(res, :use_cuda)) != $(getproperty(ref, :use_cuda)))"
        )
        string(getproperty(res, :real_type)) == string(getproperty(ref, :real_type)) || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: real_type differs ($(getproperty(res, :real_type)) != $(getproperty(ref, :real_type)))"
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
This helper sorts the supplied scalar step sizes and checks adjacent values for
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
    atol = 1e-12,
)
    T = eltype(hs)
    atolT = convert(T, atol)

    p = sortperm(hs)
    hs_sorted = hs[p]

    for (h_prev, h_cur) in zip(hs_sorted, Iterators.drop(hs_sorted, 1))
        isapprox(h_cur, h_prev; atol = atolT, rtol = zero(T)) &&
            JobLoggerTools.error_benji(
                "Duplicate h detected during merge: h=$(h_cur)"
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
