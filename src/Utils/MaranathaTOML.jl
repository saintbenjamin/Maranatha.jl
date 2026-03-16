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

    return (
        toml_path      = abs_toml_path,
        integrand_file = integrand_file,
        integrand_name = integrand_name,

        a = Float64(domain_section["a"]),
        b = Float64(domain_section["b"]),
        dim = Int(get(domain_section, "dim", 1)),

        nsamples = Int.(sampling_section["nsamples"]),

        rule = Symbol(quadrature_section["rule"]),
        boundary = Symbol(quadrature_section["boundary"]),

        err_method = Symbol(get(error_section, "err_method", "refinement")),
        fit_terms  = Int(get(error_section, "fit_terms", 4)),
        nerr_terms = Int(get(error_section, "nerr_terms", 3)),
        ff_shift   = Int(get(error_section, "ff_shift", 0)),

        use_error_jet = Bool(get(execution_section, "use_error_jet", false)),

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

# Arguments
- `cfg`: Normalized configuration bundle, typically produced by
  [`parse_run_config_from_toml`](@ref).

# Returns
- `Nothing`.

# Errors
- Throws if `a >= b`.
- Throws if `dim < 1`.
- Throws if `nsamples` is empty or contains invalid entries.
- Throws if the integrand file does not exist.
- Throws if `fit_terms < 1`, `nerr_terms < 1`, or `ff_shift < 0`.
- Throws if `err_method` is not contained in [`VALID_ERR_METHODS`](@ref).

# Notes
- This helper does not verify whether the loaded integrand signature is
  compatible with the requested dimensionality.
"""
function validate_run_config(cfg)::Nothing
    cfg.a < cfg.b || error(
        "Invalid domain: require a < b, but got a=$(cfg.a), b=$(cfg.b)."
    )

    cfg.dim >= 1 || error(
        "Invalid dim: dim must be >= 1, but got dim=$(cfg.dim)."
    )

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

    return nothing
end

end  # module MaranathaTOML