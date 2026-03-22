# ============================================================================
# src/Utils/MaranathaTOML/api/validate_run_config.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
