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

Set of supported error-estimation method identifiers accepted by
[`Maranatha.Runner.run_Maranatha`](@ref).

# Description

This constant enumerates the allowed values of the `err_method`
configuration parameter used during numerical integration.

The value is validated by [`validate_run_config`](@ref) when executing
TOML-driven runs and is forwarded to the internal error-estimation
dispatch system.

# Supported methods

The currently supported identifiers are

* `:forwarddiff`
* `:taylorseries`
* `:enzyme`
* `:fastdifferentiation`

# Notes

The values correspond to derivative backends used by the error
estimation subsystem.

Users should supply one of these symbols when configuring runs via

* direct `run_Maranatha(...; err_method=...)` calls
* TOML configuration files parsed by [`parse_run_config_from_toml`](@ref)
"""
const VALID_ERR_METHODS = Set([
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

Load a user-defined integrand function from a Julia source file in an isolated module.

# Function description

This helper evaluates a Julia source file inside a freshly created module and
extracts the function named by `func_name`.

The purpose of the isolated module is to avoid polluting the main `Maranatha`
namespace with user-defined globals or helper symbols appearing inside the
integrand file.

# Arguments

`path::AbstractString`
: Path to the Julia source file containing the user-defined integrand.

# Keyword arguments

`func_name::Symbol = :integrand`
: Name of the function to retrieve from the loaded file.

# Returns

The extracted integrand function.

# Errors

* Throws an error if the file does not exist.
* Throws an error if `func_name` is not defined in the loaded file.
* Throws an error if the retrieved object exists but is not a function.

# Notes

The file is executed as Julia code via `Base.include`.  This mechanism is
therefore intended for trusted local user files.
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

Parse a Maranatha TOML configuration file into a normalized run configuration.

# Function description

This helper reads a TOML file, extracts the supported configuration sections,
normalizes path-like entries, and converts selected string-valued options into
the forms expected by [`Maranatha.Runner.run_Maranatha`](@ref).

In particular, relative paths such as the integrand file path and save path are
interpreted relative to the TOML file location rather than the current working
directory.

# Arguments

`toml_path::AbstractString`
: Path to the TOML configuration file.

# Returns

A normalized configuration `NamedTuple` containing fields such as

* `integrand_file`
* `integrand_name`
* `a`, `b`, `dim`
* `nsamples`
* `rule`, `boundary`
* `err_method`, `fit_terms`, `nerr_terms`, `ff_shift`
* `use_threads`
* `name_prefix`, `save_path`, `write_summary`, `save_file`

# Errors

* Throws an error if the TOML file does not exist.
* Throws an error if required fields such as `[integrand].file`,
  `[domain].a`, `[domain].b`, `[sampling].nsamples`,
  `[quadrature].rule`, or `[quadrature].boundary` are missing.

# Notes

This routine performs parsing and normalization only.  Semantic validation of
the resulting configuration is deferred to [`validate_run_config`](@ref).
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

        err_method = Symbol(get(error_section, "err_method", "forwarddiff")),
        fit_terms  = Int(get(error_section, "fit_terms", 2)),
        nerr_terms = Int(get(error_section, "nerr_terms", 1)),
        ff_shift   = Int(get(error_section, "ff_shift", 0)),

        use_threads = Bool(get(execution_section, "use_threads", false)),

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

This helper checks whether a parsed TOML configuration is structurally and
numerically suitable for execution by [`Maranatha.Runner.run_Maranatha`](@ref).

The validation is intentionally limited to conditions that can be checked
reliably without executing the user-defined integrand itself.

# Arguments

`cfg`
: Normalized configuration bundle, typically produced by
  [`parse_run_config_from_toml`](@ref).

# Returns

Nothing.

# Errors

* Throws an error if the integration domain does not satisfy `a < b`.
* Throws an error if `dim < 1`.
* Throws an error if `nsamples` is empty, non-integer, or contains non-positive values.
* Throws an error if the integrand file does not exist.
* Throws an error if `fit_terms < 1`, `nerr_terms < 1`, or `ff_shift < 0`.
* Throws an error if `err_method` is not one of the supported methods in
  `VALID_ERR_METHODS`.

# Notes

This routine does not verify whether the loaded integrand function signature is
compatible with `dim`.  Any such mismatch is left to the later execution stage.
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