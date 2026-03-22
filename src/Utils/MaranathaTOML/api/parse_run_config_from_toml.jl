# ============================================================================
# src/Utils/MaranathaTOML/api/parse_run_config_from_toml.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
