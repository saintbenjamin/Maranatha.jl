# ============================================================================
# src/Utils/MaranathaTOML.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module MaranathaTOML

TOML parsing and validation helpers for `Maranatha.jl` run configurations.

# Module description
`Maranatha.Utils.MaranathaTOML` turns user-facing TOML files into normalized
configuration bundles, validates them, and loads integrand functions from local
Julia source files.

It supports scalar and axis-wise domain / rule / boundary specifications and
enforces the current refinement restriction that axis-wise rules must belong to
one common family.

# Main entry points
- [`parse_run_config_from_toml`](@ref)
- [`validate_run_config`](@ref)
- [`load_integrand_from_file`](@ref)
"""
module MaranathaTOML

import ..TOML
import ..DoubleFloats

import ..QuadratureBoundarySpec

"""
    VALID_ERR_METHODS :: Set{Symbol}

Supported `err_method` selectors for TOML-driven runs.

# Description
This constant contains the symbol values currently accepted for
`cfg.err_method` by [`validate_run_config`](@ref).

The set is used only for configuration validation. It does not perform
backend dispatch by itself.

# Notes
- Supported values are `:refinement`, `:forwarddiff`, `:taylorseries`,
  `:enzyme`, and `:fastdifferentiation`.
- Derivative-based methods and `:refinement` use different `nerr_terms`
  validation rules; see [`validate_run_config`](@ref).
"""
const VALID_ERR_METHODS = Set([
    :refinement,
    :forwarddiff,
    :taylorseries,
    :enzyme,
    :fastdifferentiation,
])

"""
    _is_domain_collection(x) -> Bool

Return `true` if `x` is treated as a collection-valued domain endpoint.

# Function description

This predicate is the internal domain-style classifier used by the TOML
parsing and validation pipeline.

The current implementation returns `true` only for:

- `Tuple`
- `AbstractVector`

All other values are treated as scalar endpoints and return `false`.

# Arguments

- `x`: Domain endpoint candidate.

# Returns

- `Bool`: `true` if `x` is a tuple or vector, `false` otherwise.
"""
@inline function _is_domain_collection(x)
    return x isa Tuple || x isa AbstractVector
end

"""
    _normalize_domain_endpoint(x)

Normalize a domain endpoint into the internal storage form used by this module.

# Function description

This helper converts tuple- and vector-like endpoints into a freshly
allocated `Vector` via `collect`. Scalar values are returned unchanged.

The result is used by [`parse_run_config_from_toml`](@ref) so that
collection-valued endpoints are stored uniformly before later validation.
# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).

# Returns

- `collect(x)` if `x isa Tuple`
- `collect(x)` if `x isa AbstractVector`
- `x` otherwise
"""
@inline function _normalize_domain_endpoint(x)
    if x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return x
    end
end

"""
    _real_type_symbol_to_type(sym::Symbol) -> DataType

Map a supported `real_type` selector symbol to its concrete Julia scalar type.

# Function description

This helper resolves the symbolic `real_type` value used by the TOML
configuration layer into the concrete scalar type used for domain parsing.

It is currently used by [`parse_run_config_from_toml`](@ref) so that
string-valued domain literals can be parsed directly in the precision selected
by `[execution].real_type`.

# Arguments

- `sym::Symbol`: Scalar-type selector such as `:Float32`, `:Float64`,
  `:BigFloat`, or `:Double64`.

# Returns

- `DataType`: Concrete Julia scalar type corresponding to `sym`.

# Errors

- Throws if `sym` is not one of the supported `real_type` selectors.

# Notes

- `:Double64` is resolved to `DoubleFloats.Double64`.
- This helper performs selector resolution only; it does not parse values.
"""
@inline function _real_type_symbol_to_type(sym::Symbol)
    sym === :Float32  && return Float32
    sym === :Float64  && return Float64
    sym === :BigFloat && return BigFloat
    sym === :Double64 && return DoubleFloats.Double64
    error("Unsupported real_type=$(sym)")
end

"""
    _parse_domain_scalar(x, T)

Parse or convert one scalar domain endpoint into the target scalar type `T`.

# Function description

This helper is the scalar-level conversion routine used by
[`parse_run_config_from_toml`](@ref) for `[domain].a` and `[domain].b`.

Behavior depends on the input form:

- if `x isa AbstractString`, the string is stripped, parsed first as
  `BigFloat`, and then converted to `T`;
- otherwise, the value is converted directly via `T(x)`.

This allows string-valued TOML domain literals to preserve precision until the
selected `real_type` is known.

# Arguments

- `x`: Scalar domain value, typically either a numeric TOML value or a string
  literal containing a numeric value.
- `T`: Target scalar type.

# Returns

- Scalar endpoint represented in type `T`.

# Errors

- Throws if a string input cannot be parsed as a number.
- Throws if conversion to `T` fails.

# Notes

- This helper handles scalar values only. Collection-valued endpoints are
  handled by [`_parse_domain_endpoint`](@ref).
"""
@inline function _parse_domain_scalar(x, T)
    if x isa AbstractString
        return T(parse(BigFloat, strip(x)))
    else
        return T(x)
    end
end

"""
    _parse_domain_endpoint(x, T)

Parse a scalar or collection-valued domain endpoint into the target scalar type `T`.

# Function description

This helper lifts [`_parse_domain_scalar`](@ref) to the endpoint forms accepted
by the TOML configuration layer.

Behavior depends on `x`:

- if `x isa Tuple`, each element is parsed / converted and the result is
  returned as a new vector;
- if `x isa AbstractVector`, each element is parsed / converted and the result
  is returned as a new vector;
- otherwise, `x` is treated as a scalar endpoint and converted directly.

# Arguments

- `x`: Domain endpoint given either as a scalar value or an array-like
  collection of per-axis bounds.
- `T`: Target scalar type.

# Returns

- Parsed scalar `T` for scalar input.
- `Vector{T}` for tuple- or vector-like input.

# Errors

- Propagates parsing or conversion errors from [`_parse_domain_scalar`](@ref).

# Notes

- This helper accepts both numeric TOML values and string-valued numeric
  literals.
- Tuple input is normalized to a vector so downstream endpoint handling can use
  one collection representation.
"""
@inline function _parse_domain_endpoint(x, T)
    if x isa Tuple
        return [_parse_domain_scalar(v, T) for v in x]
    elseif x isa AbstractVector
        return [_parse_domain_scalar(v, T) for v in x]
    else
        return _parse_domain_scalar(x, T)
    end
end

"""
    _parse_boundary_entry(x) -> Symbol

Parse one scalar boundary entry from TOML input.

# Function description
This helper accepts either a `Symbol` or a string-like TOML value, normalizes
it to `Symbol`, and validates it against [`QuadratureBoundarySpec._decode_boundary`](@ref).

# Arguments
- `x`: Scalar boundary entry from TOML input.

# Returns
- `Symbol`: Validated boundary symbol.

# Errors
- Throws if `x` is neither a symbol nor a string.
- Throws if the decoded boundary symbol is unsupported.
"""
@inline function _parse_boundary_entry(x)::Symbol
    if x isa Symbol
        QuadratureBoundarySpec._decode_boundary(x)
        return x
    elseif x isa AbstractString
        s = Symbol(strip(x))
        QuadratureBoundarySpec._decode_boundary(s)
        return s
    else
        error("Invalid boundary entry: expected Symbol or String, got $(typeof(x)).")
    end
end

"""
    _parse_boundary_spec(x)

Parse a scalar or axis-wise boundary specification from TOML input.

# Function description
Scalar input is parsed as one boundary entry. Tuple/vector input is parsed
entrywise and returned as a tuple of boundary symbols.

# Arguments
- `x`: TOML boundary value, scalar or array-like.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Propagates entry-level parsing errors from [`_parse_boundary_entry`](@ref).
"""
@inline function _parse_boundary_spec(x)
    if x isa Tuple
        vals = Tuple(_parse_boundary_entry(v) for v in x)
        return vals
    elseif x isa AbstractVector
        vals = Tuple(_parse_boundary_entry(v) for v in x)
        return vals
    else
        return _parse_boundary_entry(x)
    end
end

"""
    _parse_rule_entry(x) -> Symbol

Parse one scalar quadrature-rule entry from TOML input.

# Function description
This helper accepts either a `Symbol` or a string-like TOML value and
normalizes it to `Symbol`.

# Arguments
- `x`: Scalar rule entry from TOML input.

# Returns
- `Symbol`: Parsed rule symbol.

# Errors
- Throws if `x` is neither a symbol nor a string.
"""
@inline function _parse_rule_entry(x)::Symbol
    if x isa Symbol
        return x
    elseif x isa AbstractString
        return Symbol(strip(x))
    else
        error("Invalid rule entry: expected Symbol or String, got $(typeof(x)).")
    end
end

"""
    _parse_rule_spec(x)

Parse a scalar or axis-wise quadrature-rule specification from TOML input.

# Function description
Scalar input is parsed as one rule symbol. Tuple/vector input is parsed
entrywise and returned as a tuple of rule symbols.

# Arguments
- `x`: TOML rule value, scalar or array-like.

# Returns
- Scalar `Symbol` or tuple of `Symbol` values.

# Errors
- Propagates entry-level parsing errors from [`_parse_rule_entry`](@ref).
"""
@inline function _parse_rule_spec(x)
    if x isa Tuple
        return Tuple(_parse_rule_entry(v) for v in x)
    elseif x isa AbstractVector
        return Tuple(_parse_rule_entry(v) for v in x)
    else
        return _parse_rule_entry(x)
    end
end

"""
    _rule_family_local(rule::Symbol) -> Symbol

Classify a TOML-parsed scalar rule symbol by family.

# Function description
This helper performs a lightweight local family classification used during TOML
validation, without depending on the quadrature module's internal helpers.

# Arguments
- `rule::Symbol`: Scalar rule symbol to classify.

# Returns
- `Symbol`: One of `:newton_cotes`, `:gauss`, or `:bspline`.

# Errors
- Throws if `rule` is not a supported rule symbol.
"""
@inline function _rule_family_local(rule::Symbol)::Symbol
    s = String(rule)

    startswith(s, "newton_p") && return :newton_cotes
    startswith(s, "gauss_p") && return :gauss

    if startswith(s, "bspline_interp_p") || startswith(s, "bspline_smooth_p")
        return :bspline
    end

    error("Invalid rule specification: unsupported rule symbol $(rule).")
end

"""
    _rule_at_local(rule, d::Int, dim::Int) -> Symbol

Resolve the scalar TOML-parsed rule symbol used on axis `d`.

# Function description
Scalar rules are returned unchanged after validation. Tuple/vector rules are
validated against `dim`, indexed at `d`, and checked for supported family
membership.

# Arguments
- `rule`: Scalar or axis-wise TOML-parsed rule specification.
- `d::Int`: Axis index to resolve.
- `dim::Int`: Expected axis count for axis-wise input.

# Returns
- `Symbol`: Scalar rule symbol used on axis `d`.

# Errors
- Throws if the rule specification is malformed or unsupported.
"""
@inline function _rule_at_local(rule, d::Int, dim::Int)::Symbol
    if rule isa Symbol
        _rule_family_local(rule)
        return rule
    elseif rule isa Tuple || rule isa AbstractVector
        length(rule) == dim || error(
            "Invalid rule specification: length(rule) must equal dim=$(dim)."
        )
        rd = rule[d]
        rd isa Symbol || error(
            "Invalid rule specification: rule[$d] must be a Symbol."
        )
        _rule_family_local(rd)
        return rd
    else
        error(
            "Invalid rule specification: expected Symbol or tuple/vector of Symbols, got $(typeof(rule))."
        )
    end
end

"""
    _validate_rule_spec_local(rule, dim::Int) -> Nothing

Validate that a TOML-parsed rule specification is well formed for dimension
`dim`.

# Function description
This helper accepts either a scalar rule symbol shared across all axes or a
tuple/vector of per-axis rule symbols of length `dim`. Every resolved axis
entry is checked for supported family membership.

# Arguments
- `rule`: TOML-parsed rule specification.
- `dim::Int`: Expected problem dimension.

# Returns
- `nothing`

# Errors
- Throws if `dim < 1`, if an axis-wise rule specification has the wrong
  length, or if any axis-local rule is unsupported.
"""
@inline function _validate_rule_spec_local(rule, dim::Int)::Nothing
    dim >= 1 || error("Invalid dim: dim must be >= 1, but got dim=$(dim).")

    for d in 1:dim
        _rule_at_local(rule, d, dim)
    end

    return nothing
end

"""
    _validate_refinement_rule_family_local(
        rule,
        dim::Int,
        err_method::Symbol,
    ) -> Nothing

Validate the current refinement restriction on axis-wise TOML rule specs.

# Function description
When `err_method != :refinement`, this helper returns immediately. For
refinement runs, it verifies that all axis-local rule entries belong to one
common quadrature family.

# Arguments
- `rule`: TOML-parsed rule specification.
- `dim::Int`: Expected problem dimension.
- `err_method::Symbol`: Parsed error-method selector.

# Returns
- `nothing`

# Errors
- Throws if `err_method == :refinement` and the resolved per-axis rules do not
  all belong to one family.
- Propagates rule-validation errors from [`_rule_at_local`](@ref) and
  [`_rule_family_local`](@ref).
"""
@inline function _validate_refinement_rule_family_local(
    rule,
    dim::Int,
    err_method::Symbol,
)::Nothing
    err_method === :refinement || return nothing

    fam = _rule_family_local(_rule_at_local(rule, 1, dim))
    for d in 2:dim
        fam_d = _rule_family_local(_rule_at_local(rule, d, dim))
        fam_d == fam || error(
            "Invalid rule specification: err_method=:refinement requires all axis-wise rules to belong to one family, but got rule=$(rule)."
        )
    end

    return nothing
end

"""
    _domain_axis_values(
        x, 
        dim::Int
    ) -> Vector

Expand a domain endpoint into explicit per-axis values of length `dim`.

# Function description

This helper converts a scalar or collection-valued endpoint into the
axis-by-axis representation used by [`validate_run_config`](@ref).

Behavior depends on `x`:

- if `x isa Tuple`, the tuple length must equal `dim` and the elements are
  copied into a new vector;
- if `x isa AbstractVector`, the vector length must equal `dim` and the
  values are collected into a new vector;
- otherwise, `x` is treated as a scalar and replicated `dim` times.

# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).
- `dim`: Expected number of dimensions.

# Returns

- `Vector`: A length-`dim` vector of axis endpoint values.

# Errors

- Throws an error if a tuple or vector input does not have length `dim`.
"""
@inline function _domain_axis_values(
    x, 
    dim::Int
)
    if x isa Tuple
        length(x) == dim || error(
            "Domain tuple length mismatch: expected dim=$(dim), got length=$(length(x))."
        )
        return [x[i] for i in 1:dim]
    elseif x isa AbstractVector
        length(x) == dim || error(
            "Domain vector length mismatch: expected dim=$(dim), got length=$(length(x))."
        )
        return collect(x)
    else
        return fill(x, dim)
    end
end

"""
    load_integrand_from_file(
        path::AbstractString;
        func_name::Symbol = :integrand
    ) -> Function

Load a named function binding from a Julia source file.

# Function description
This helper resolves `path` to an absolute path, evaluates the file in a
fresh temporary module, and then retrieves the binding named by `func_name`.

Using an isolated module keeps helper symbols defined in the loaded file out
of the main package namespace.

# Arguments
- `path::AbstractString`: Path to the Julia source file to load.

# Keyword arguments
- `func_name::Symbol`: Name of the function to retrieve from the loaded file.

# Returns
- `Function`: Loaded integrand function.

# Errors
- Throws if the file does not exist.
- Throws if `func_name` is not defined in the loaded file.
- Throws if the retrieved binding exists but is not a function.

# Notes
- The file is executed with `Base.include` inside a `gensym`-named module.
- The returned value is the binding itself; this helper does not call it.
- This mechanism is intended for trusted local Julia source files.
"""
function load_integrand_from_file(
    path::AbstractString;
    func_name::Symbol = :integrand
)::Function
    abs_path = abspath(path)

    isfile(abs_path) || error("Integrand file not found: $abs_path")

    mod = Module(gensym(:MaranathaUserIntegrand))
    Base.include(mod, abs_path)

    isdefined(mod, func_name) || error(
        "Integrand file does not define function `$(func_name)`: $abs_path"
    )

    f = Base.invokelatest(getproperty, mod, func_name)

    f isa Function || error(
        "`$(func_name)` exists but is not a function: $abs_path"
    )

    return f
end

"""
    parse_run_config_from_toml(
        toml_path::AbstractString
    ) -> NamedTuple

Parse a Maranatha [`TOML`](https://toml.io/en/) file into a normalized run configuration.

# Function description
This helper reads a [`TOML`](https://toml.io/en/) file, extracts the supported
configuration sections, normalizes selected values, and returns a `NamedTuple`
consumed by the run pipeline.

In particular:

- relative paths such as `[integrand].file` and `[output].save_path` are
  resolved relative to the TOML file location;
- string-like selectors such as `rule`, `boundary`, `err_method`, and
  `real_type` are converted to `Symbol` values;
- `rule` and `boundary` may each be supplied either as scalar values or as
  arrays describing axis-wise specifications;
- domain endpoints `[domain].a` and `[domain].b` are parsed in the scalar type
  selected by `[execution].real_type`;
- domain endpoints may be supplied either as ordinary numeric TOML values or as
  string-valued numeric literals, in either scalar or array form;
- collection-valued endpoints are normalized to concrete vectors for downstream
  validation.

This helper also performs a small amount of structural checking before
returning:

- `[domain].a` and `[domain].b` must both be scalars or both be collections;
- `[sampling].nsamples` must be an array-like value.

# Arguments
- `toml_path::AbstractString`: Path to the [`TOML`](https://toml.io/en/) configuration file.

# Returns
- `NamedTuple`: Normalized run-configuration bundle with fields
  `toml_path`, `integrand_file`, `integrand_name`, `a`, `b`, `dim`,
  `nsamples`, `rule`, `boundary`, `err_method`, `fit_terms`,
  `nerr_terms`, `ff_shift`, `use_error_jet`, `use_cuda`, `real_type`,
  `name_prefix`, `save_path`, `write_summary`, and `save_file`.

# Errors
- Throws if the [`TOML`](https://toml.io/en/) file does not exist.
- Throws if required fields such as `[integrand].file`, `[domain].a`,
  `[domain].b`, `[sampling].nsamples`, `[quadrature].rule`, or
  `[quadrature].boundary` are missing.
- Throws if `[execution].real_type` is unsupported.
- Throws if a string-valued domain literal cannot be parsed as a number.
- Throws if `[domain].a` and `[domain].b` mix scalar and collection styles.
- Throws if `[sampling].nsamples` is not an array-like value.

# Notes
- This helper performs parsing and normalization only.
- Semantic validation is deferred to [`validate_run_config`](@ref).
- Domain endpoints are converted to the scalar type selected by
  `[execution].real_type` during parsing.
- Rule specifications are validated locally during parsing so that unsupported
  rule symbols and mixed-family refinement rule sets fail early.
- This allows wizard-generated string-valued domain literals to preserve
  precision until parse time while remaining backward compatible with ordinary
  numeric [`TOML`](https://toml.io/en/) input.
- If `[error].err_method` is omitted, it defaults to `:refinement`,
  which activates the resolution-refinement error estimator.
- If `[integrand].name` is omitted, it defaults to `:integrand`.
- If `[domain].dim` is omitted, it defaults to `1`.
- If `[execution].real_type` is omitted, it defaults to `:Float64`.
- If `[execution].use_error_jet` is omitted, it defaults to `false`.
- If `[execution].use_cuda` is omitted, it defaults to `false`.
- If `[output].name_prefix` is omitted, it defaults to `"Maranatha"`.
- If `[output].save_path` is omitted, it defaults to the TOML file directory.
- If `[output].write_summary` is omitted, it defaults to `true`.
- If `[output].save_file` is omitted, it defaults to `true`.
- If `[error].fit_terms`, `[error].nerr_terms`, or `[error].ff_shift`
  are omitted, they default to `4`, `3`, and `0`, respectively.
- Rectangular-domain TOML input is supported by allowing `[domain].a` and
  `[domain].b` to be arrays of length `dim`.
"""
function parse_run_config_from_toml(
    toml_path::AbstractString
)
    abs_toml_path = abspath(toml_path)
    isfile(abs_toml_path) || error("TOML configuration file not found: $abs_toml_path")

    cfg = TOML.parsefile(abs_toml_path)
    cfg_dir = dirname(abs_toml_path)

    integrand_section  = get(cfg, "integrand", Dict{String,Any}())
    domain_section     = get(cfg, "domain", Dict{String,Any}())
    sampling_section   = get(cfg, "sampling", Dict{String,Any}())
    quadrature_section = get(cfg, "quadrature", Dict{String,Any}())
    error_section      = get(cfg, "error", Dict{String,Any}())
    execution_section  = get(cfg, "execution", Dict{String,Any}())
    output_section     = get(cfg, "output", Dict{String,Any}())

    integrand_relpath = get(integrand_section, "file", nothing)
    integrand_relpath === nothing && error(
        "Missing required field `[integrand].file` in TOML: $abs_toml_path"
    )

    integrand_name = Symbol(get(integrand_section, "name", "integrand"))

    haskey(domain_section, "a") || error(
        "Missing required field `[domain].a` in TOML: $abs_toml_path"
    )
    haskey(domain_section, "b") || error(
        "Missing required field `[domain].b` in TOML: $abs_toml_path"
    )
    haskey(sampling_section, "nsamples") || error(
        "Missing required field `[sampling].nsamples` in TOML: $abs_toml_path"
    )
    haskey(quadrature_section, "rule") || error(
        "Missing required field `[quadrature].rule` in TOML: $abs_toml_path"
    )
    haskey(quadrature_section, "boundary") || error(
        "Missing required field `[quadrature].boundary` in TOML: $abs_toml_path"
    )

    integrand_file = isabspath(integrand_relpath) ?
        normpath(integrand_relpath) :
        normpath(joinpath(cfg_dir, integrand_relpath))

    save_path_raw = String(get(output_section, "save_path", "."))
    save_path = isabspath(save_path_raw) ?
        normpath(save_path_raw) :
        normpath(joinpath(cfg_dir, save_path_raw))

    real_type = get(execution_section, "real_type", "Float64")
    real_type = real_type isa Symbol ? real_type : Symbol(String(real_type))
    T_domain = _real_type_symbol_to_type(real_type)

    a_raw = _normalize_domain_endpoint(
        _parse_domain_endpoint(domain_section["a"], T_domain)
    )
    b_raw = _normalize_domain_endpoint(
        _parse_domain_endpoint(domain_section["b"], T_domain)
    )

    if _is_domain_collection(a_raw) != _is_domain_collection(b_raw)
        error(
            "Invalid TOML domain specification: `[domain].a` and `[domain].b` " *
            "must both be scalars, or both be arrays."
        )
    end

    raw_nsamples = sampling_section["nsamples"]
    raw_nsamples isa AbstractVector || error(
        "Invalid `[sampling].nsamples` in TOML: expected an array, got $(typeof(raw_nsamples))."
    )

    dim_val = Int(get(domain_section, "dim", 1))

    rule_parsed = _parse_rule_spec(quadrature_section["rule"])
    boundary_parsed = _parse_boundary_spec(quadrature_section["boundary"])
    err_method_parsed = Symbol(get(error_section, "err_method", "refinement"))

    _validate_rule_spec_local(rule_parsed, dim_val)
    _validate_refinement_rule_family_local(rule_parsed, dim_val, err_method_parsed)
    QuadratureBoundarySpec._validate_boundary_spec(boundary_parsed, dim_val)

    return (
        toml_path      = abs_toml_path,
        integrand_file = integrand_file,
        integrand_name = integrand_name,

        a = a_raw,
        b = b_raw,
        dim = Int(get(domain_section, "dim", 1)),

        nsamples = collect(Int.(raw_nsamples)),

        rule = rule_parsed,
        boundary = boundary_parsed,

        err_method = err_method_parsed,
        fit_terms  = Int(get(error_section, "fit_terms", 4)),
        nerr_terms = Int(get(error_section, "nerr_terms", 3)),
        ff_shift   = Int(get(error_section, "ff_shift", 0)),

        use_error_jet = Bool(get(execution_section, "use_error_jet", false)),
        use_cuda      = Bool(get(execution_section, "use_cuda", false)),
        real_type     = real_type,

        name_prefix   = String(get(output_section, "name_prefix", "Maranatha")),
        save_path     = save_path,
        write_summary = Bool(get(output_section, "write_summary", true)),
        save_file     = Bool(get(output_section, "save_file", true)),
    )
end

"""
    validate_run_config(cfg) -> Nothing

Validate a normalized Maranatha run configuration.

# Function description
This helper checks whether a parsed or manually constructed configuration is
structurally and numerically suitable for execution.

The validation is intentionally limited to conditions that can be checked
reliably without executing the user-defined integrand itself.

It supports both isotropic and rectangular-domain configurations:

- if `cfg.a` and `cfg.b` are scalars, the same interval is used on every axis;
- if `cfg.a` and `cfg.b` are tuple/vector-like collections, they are interpreted
  as per-axis bounds and must match `cfg.dim`.

# Arguments
- `cfg`: Normalized configuration bundle, typically produced by
  [`parse_run_config_from_toml`](@ref).

# Returns
- `Nothing`.

# Errors
- Throws if `dim < 1`.
- Throws if tuple- or vector-like domain endpoints do not have length `dim`.
- Throws if any axis fails the strict bound check `a[i] < b[i]`.
- Throws if scalar / collection domain styles are mixed between `a` and `b`.
- Throws if `nsamples` is empty or contains invalid entries.
- Throws if `nsamples` contains duplicate entries.
- Throws if the integrand file does not exist.
- Throws if `fit_terms < 1`.
- Throws if `err_method == :refinement` and `nerr_terms < 0`.
- Throws if `err_method != :refinement` and `nerr_terms < 1`.
- Throws if `ff_shift < 0`.
- Throws if `err_method` is not contained in [`VALID_ERR_METHODS`](@ref).
- Throws if `real_type` is not one of the supported scalar-type selectors.
- Throws if `use_cuda == true` but `real_type` is not CUDA-compatible.
- Throws if an axis-wise `rule` specification has the wrong length or, for
  `err_method == :refinement`, mixes multiple quadrature families.

# Notes
- This helper does not verify whether the loaded integrand signature is
  compatible with the requested dimensionality.
- CUDA mode currently supports only `:Float32` and `:Float64`.
- Supported `real_type` selectors are `:Float32`, `:Float64`, `:Double64`,
  and `:BigFloat`.
- For rectangular domains, endpoint-length consistency is checked through
  [`_domain_axis_values`](@ref).
- Axis-wise `boundary` specifications are validated through
  [`QuadratureBoundarySpec._validate_boundary_spec`](@ref).
"""
function validate_run_config(cfg)::Nothing
    cfg.dim >= 1 || error(
        "Invalid dim: dim must be >= 1, but got dim=$(cfg.dim)."
    )

    a_axes = _domain_axis_values(cfg.a, cfg.dim)
    b_axes = _domain_axis_values(cfg.b, cfg.dim)
    QuadratureBoundarySpec._validate_boundary_spec(cfg.boundary, cfg.dim)
    _validate_rule_spec_local(cfg.rule, cfg.dim)
    _validate_refinement_rule_family_local(cfg.rule, cfg.dim, cfg.err_method)

    for i in 1:cfg.dim
        a_axes[i] < b_axes[i] || error(
            "Invalid domain on axis $i: require a[$i] < b[$i], " *
            "but got a[$i]=$(a_axes[i]), b[$i]=$(b_axes[i])."
        )
    end

    if _is_domain_collection(cfg.a) != _is_domain_collection(cfg.b)
        error(
            "Invalid domain specification: `a` and `b` must both be scalars, " *
            "or both be tuple/vector-like collections."
        )
    end

    !isempty(cfg.nsamples) || error(
        "Invalid nsamples: the list must not be empty."
    )

    length(unique(cfg.nsamples)) == length(cfg.nsamples) || error(
        "Invalid nsamples: duplicate entries are not allowed, but got $(cfg.nsamples)."
    )

    all(n -> n isa Integer, cfg.nsamples) || error(
        "Invalid nsamples: all entries must be integers."
    )

    all(n -> n > 0, cfg.nsamples) || error(
        "Invalid nsamples: all entries must be positive."
    )

    isfile(cfg.integrand_file) || error(
        "Integrand file not found: $(cfg.integrand_file)"
    )

    cfg.fit_terms >= 1 || error(
        "Invalid fit_terms: must be >= 1, but got $(cfg.fit_terms)."
    )

    if cfg.err_method == :refinement
        cfg.nerr_terms >= 0 || error(
            "Invalid nerr_terms: for err_method=:refinement, nerr_terms must be >= 0, but got $(cfg.nerr_terms)."
        )
    else
        cfg.nerr_terms >= 1 || error(
            "Invalid nerr_terms: for derivative-based error methods, nerr_terms must be >= 1, but got $(cfg.nerr_terms)."
        )
    end

    cfg.ff_shift >= 0 || error(
        "Invalid ff_shift: must be >= 0, but got $(cfg.ff_shift)."
    )

    cfg.err_method in VALID_ERR_METHODS || error(
        "Unsupported err_method: $(cfg.err_method). Supported values are " *
        "$(collect(VALID_ERR_METHODS))."
    )

    cfg.real_type in (:Float32, :Float64, :Double64, :BigFloat) || error(
        "Unsupported real_type: $(cfg.real_type). Supported values are " *
        "[:Float32, :Float64, :Double64, :BigFloat]."
    )

    if cfg.use_cuda
        cfg.real_type in (:Float32, :Float64) || error(
            "CUDA mode requires real_type to be :Float32 or :Float64, but got $(cfg.real_type)."
        )
    end

    return nothing
end

end  # module MaranathaTOML
