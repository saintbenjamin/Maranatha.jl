# ============================================================================
# src/Utils/MaranathaTOML.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MaranathaTOML

import ..TOML
import ...DoubleFloats

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

    return (
        toml_path      = abs_toml_path,
        integrand_file = integrand_file,
        integrand_name = integrand_name,

        a = a_raw,
        b = b_raw,
        dim = Int(get(domain_section, "dim", 1)),

        nsamples = collect(Int.(raw_nsamples)),

        rule = Symbol(quadrature_section["rule"]),
        boundary = Symbol(quadrature_section["boundary"]),

        err_method = Symbol(get(error_section, "err_method", "refinement")),
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

# Notes
- This helper does not verify whether the loaded integrand signature is
  compatible with the requested dimensionality.
- CUDA mode currently supports only `:Float32` and `:Float64`.
- Supported `real_type` selectors are `:Float32`, `:Float64`, `:Double64`,
  and `:BigFloat`.
- For rectangular domains, endpoint-length consistency is checked through
  [`_domain_axis_values`](@ref).
"""
function validate_run_config(cfg)::Nothing
    cfg.dim >= 1 || error(
        "Invalid dim: dim must be >= 1, but got dim=$(cfg.dim)."
    )

    a_axes = _domain_axis_values(cfg.a, cfg.dim)
    b_axes = _domain_axis_values(cfg.b, cfg.dim)

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