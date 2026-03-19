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

"""
    VALID_ERR_METHODS :: Set{Symbol}

Set of supported error-method identifiers for TOML-driven runs.

# Description
This constant enumerates the allowed values of the `err_method`
configuration parameter accepted by the run-configuration pipeline.

It is used by [`validate_run_config`](@ref) to reject unsupported error-method
selections before execution begins.

# Notes
- The values correspond to error-estimation backends used by the
  Maranatha error-estimation layer, including both derivative-based
  and refinement-based methods.
- The constant is intended for validation, not for backend dispatch by itself.
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

Return `true` if `x` represents a multi-component domain endpoint.

# Function description

This predicate checks whether a domain value encodes multiple coordinates,
such as a tuple or vector used to specify per-axis bounds in a
multi-dimensional integration domain.

Scalar values return `false`, while tuple- or vector-like values return `true`.

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

Normalize a domain endpoint into a storable container form.

# Function description

This helper converts tuple- or vector-like endpoints into a concrete
`Vector`, ensuring a consistent mutable container representation for
downstream processing or serialization. Scalar values are returned
unchanged.

The normalization avoids ambiguity between tuples and arrays when
handling domain metadata.

# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).

# Returns

- A normalized value:

  - `Vector` if `x` is a tuple or vector,
  - the original value if `x` is scalar.
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
    _domain_axis_values(x, dim::Int) -> Vector

Expand a domain endpoint into explicit per-axis values.

# Function description

This routine produces a length-`dim` vector describing the endpoint value
along each coordinate axis.

Behavior depends on the input type:

- Tuple input: elements are copied into a new vector after verifying
  that the tuple length equals `dim`.
- Vector input: collected into a new vector after verifying length.
- Scalar input: replicated across all dimensions.

Length mismatches for tuple or vector inputs trigger an error, ensuring
that domain specifications remain dimensionally consistent.

# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).
- `dim`: Expected number of dimensions.

# Returns

- `Vector`: A length-`dim` vector of axis endpoint values.

# Errors

- Throws an error if a tuple or vector input does not have length `dim`.
"""
@inline function _domain_axis_values(x, dim::Int)
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

Load a user-defined integrand function from a Julia source file.

# Function description
This helper evaluates the given Julia source file inside a fresh temporary
module and retrieves the function named by `func_name`.

Using an isolated module prevents user-defined helper symbols in the integrand
file from polluting the main package namespace.

# Arguments
- `path::AbstractString`: Path to the Julia source file containing the
  user-defined integrand.

# Keyword arguments
- `func_name::Symbol`: Name of the function to retrieve from the loaded file.

# Returns
- `Function`: Loaded integrand function.

# Errors
- Throws if the file does not exist.
- Throws if `func_name` is not defined in the loaded file.
- Throws if the retrieved binding exists but is not a function.

# Notes
- The file is executed via `Base.include`.
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

Parse a Maranatha [`TOML`](https://toml.io/en/) configuration file into a normalized run configuration.

# Function description
This helper reads a [`TOML`](https://toml.io/en/) configuration file, extracts the supported sections,
normalizes path-like entries, and converts selected option values into the
forms expected by the Maranatha run pipeline.

In particular, relative paths are interpreted relative to the [`TOML`](https://toml.io/en/) file
location rather than the current working directory.

Domain endpoints are accepted in two forms:

- scalar values, representing an isotropic domain shared across all axes, or
- arrays, representing per-axis bounds for rectangular domains.

When arrays are used, they are preserved as normalized vector-like endpoint data
and later validated against `dim`.

# Arguments
- `toml_path::AbstractString`: Path to the [`TOML`](https://toml.io/en/) configuration file.

# Returns
- `NamedTuple`: Normalized run-configuration bundle.

# Errors
- Throws if the [`TOML`](https://toml.io/en/) file does not exist.
- Throws if required fields such as `[integrand].file`, `[domain].a`,
  `[domain].b`, `[sampling].nsamples`, `[quadrature].rule`, or
  `[quadrature].boundary` are missing.

# Notes
- This helper performs parsing and normalization only.
- Semantic validation is deferred to [`validate_run_config`](@ref).
- If `[error].err_method` is omitted, it defaults to `:refinement`,
  which activates the resolution-refinement error estimator.
- If `[execution].use_error_jet` is omitted, it defaults to `false`.
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

    a_raw = _normalize_domain_endpoint(domain_section["a"])
    b_raw = _normalize_domain_endpoint(domain_section["b"])

    if _is_domain_collection(a_raw) != _is_domain_collection(b_raw)
        error(
            "Invalid TOML domain specification: `[domain].a` and `[domain].b` " *
            "must both be scalars, or both be arrays."
        )
    end

    return (
        toml_path      = abs_toml_path,
        integrand_file = integrand_file,
        integrand_name = integrand_name,

        a = a_raw,
        b = b_raw,
        dim = Int(get(domain_section, "dim", 1)),

        nsamples = Int.(sampling_section["nsamples"]),

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
This helper checks whether a parsed / normalized configuration is structurally
and numerically suitable for execution.

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
- Throws if any axis fails the strict bound check `a[i] < b[i]`.
- Throws if scalar / collection domain styles are mixed between `a` and `b`.
- Throws if `dim < 1`.
- Throws if `nsamples` is empty or contains invalid entries.
- Throws if the integrand file does not exist.
- Throws if `fit_terms < 1`, `nerr_terms < 1`, or `ff_shift < 0`.
- Throws if `err_method` is not contained in [`VALID_ERR_METHODS`](@ref).
- Throws if `real_type` is not one of the supported scalar-type selectors.
- Throws if `use_cuda == true` but `real_type` is not CUDA-compatible.

# Notes
- This helper does not verify whether the loaded integrand signature is
  compatible with the requested dimensionality.
- CUDA mode currently supports only `:Float32` and `:Float64`.
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

    cfg.nerr_terms >= 1 || error(
        "Invalid nerr_terms: must be >= 1, but got $(cfg.nerr_terms)."
    )

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