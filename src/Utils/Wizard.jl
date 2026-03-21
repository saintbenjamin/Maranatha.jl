# ============================================================================
# src/Utils/Wizard.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Wizard

Interactive TOML-configuration wizard for `Maranatha.jl`.

# Module description
`Maranatha.Utils.Wizard` provides a small terminal-driven workflow that collects
run settings interactively, writes a TOML configuration file, and can generate
a matching sample integrand source file.

The wizard supports both shared scalar and axis-wise rule / boundary input for
multi-dimensional runs.

# Main entry points
- [`run_wizard`](@ref)
"""
module Wizard

"""
    _prompt(
        msg::String,
        default::AbstractString
    ) -> AbstractString

Prompt the user for a string value with a default fallback.

# Function description
This helper prints an interactive prompt of the form

    msg [default]:

reads one line from standard input, and returns either the stripped user input
or `default` as passed when the input is empty or whitespace-only.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::AbstractString`: Default value used when the user presses Enter.

# Returns
- `AbstractString`: Stripped user input, or the original `default` value.

# Errors
- Propagates I/O errors from `readline()`.

# Notes
- This is a low-level helper used by the wizard input routines.
- Only user-supplied input is normalized with `strip(...)`; `default` is not
  stripped inside this helper.
"""
function _prompt(
    msg::String, 
    default::AbstractString
)
    print("$msg [$default]: ")
    s = readline()
    isempty(strip(s)) ? default : strip(s)
end

"""
    _prompt_float(
        msg::String,
        default::Float64
    ) -> Float64

Prompt the user for a floating-point value.

# Function description
This helper calls [`_prompt`](@ref), then parses the resulting string as
`Float64`.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Float64`: Default floating-point value.

# Returns
- `Float64`: Parsed floating-point value.

# Errors
- Throws if the entered value cannot be parsed as `Float64`.
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- This helper is intended for wizard fields that are genuinely `Float64`-based.
- Precision-preserving domain input should use raw string prompting instead of
  this helper.
"""
function _prompt_float(
    msg::String, 
    default::Float64
)
    s = _prompt(msg, string(default))
    parse(Float64, s)
end

"""
    _prompt_int(
        msg::String,
        default::Int
    ) -> Int

Prompt the user for an integer value.

# Function description
This helper calls [`_prompt`](@ref), then parses the resulting string as `Int`.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Int`: Default integer value.

# Returns
- `Int`: Parsed integer value.

# Errors
- Throws if the entered value cannot be parsed as `Int`.
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- Intended for wizard fields such as `dim`, `fit_terms`, `nerr_terms`, and
  `ff_shift`.
"""
function _prompt_int(
    msg::String, 
    default::Int
)
    s = _prompt(msg, string(default))
    parse(Int, s)
end

"""
    _prompt_bool(
        msg::String,
        default::Bool
    ) -> Bool

Prompt the user for a Boolean value.

# Function description
This helper calls [`_prompt`](@ref), lowercases the resulting string, and
interprets a small accepted set of truthy values as `true`:

- `"true"`
- `"t"`
- `"1"`
- `"yes"`
- `"y"`

All other inputs are treated as `false`.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Bool`: Default Boolean value.

# Returns
- `Bool`: Interpreted Boolean value.

# Errors
- No explicit parsing errors are thrown for unrecognized inputs.
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- This helper is intentionally permissive for interactive wizard use.
"""
function _prompt_bool(
    msg::String, 
    default::Bool
)
    s = lowercase(_prompt(msg, default ? "true" : "false"))
    s in ("true","t","1","yes","y")
end

"""
    _prompt_int_vector(
        msg::String,
        default::Vector{Int}
    ) -> Vector{Int}

Prompt the user for a comma-separated list of integers.

# Function description
This helper calls [`_prompt`](@ref), splits the resulting string on commas,
and parses each field as `Int`.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Vector{Int}`: Default integer vector.

# Returns
- `Vector{Int}`: Parsed integer vector.

# Errors
- Throws if any element cannot be parsed as `Int`.
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- Intended mainly for the `[sampling].nsamples` field.
"""
function _prompt_int_vector(
    msg::String, 
    default::Vector{Int}
)
    s = _prompt(msg, join(default, ","))
    parse.(Int, strip.(split(s, ",")))
end

"""
    _prompt_float_vector(
        msg::String,
        default::Vector{Float64}
    ) -> Vector{Float64}

Prompt the user for a comma-separated list of floating-point values.

# Function description
This helper calls [`_prompt`](@ref), splits the resulting string on commas,
and parses each field as `Float64`.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Vector{Float64}`: Default floating-point vector.

# Returns
- `Vector{Float64}`: Parsed floating-point vector.

# Errors
- Throws if any element cannot be parsed as `Float64`.
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- This helper performs eager `Float64` parsing.
- It is therefore not suitable for precision-preserving domain input when the
  generated TOML should defer parsing until `real_type` is known.
"""
function _prompt_float_vector(
    msg::String,
    default::Vector{Float64}
)
    s = _prompt(msg, join(default, ","))
    parse.(Float64, strip.(split(s, ",")))
end

"""
    _prompt_literal_vector(
        msg::String,
        default::Vector{<:AbstractString}
    ) -> Vector{String}

Prompt the user for a comma-separated list of numeric literal strings.

# Function description
This helper calls [`_prompt`](@ref), splits the resulting string on commas,
strips each field, and returns the resulting strings without numeric parsing.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::Vector{<:AbstractString}`: Default literal vector.

# Returns
- `Vector{String}`: Parsed literal strings.

# Errors
- Propagates any I/O errors from [`_prompt`](@ref).

# Notes
- Intended for `[domain].a` and `[domain].b` when the wizard should preserve
  user-entered numeric text until the TOML parser later interprets it using
  `real_type`.
"""
function _prompt_literal_vector(
    msg::String,
    default::Vector{<:AbstractString}
)
    s = _prompt(msg, join(default, ","))
    String.(strip.(split(s, ",")))
end

"""
    _prompt_boundary_spec(
        msg::String,
        dim::Int;
        default_scalar::AbstractString = "LU_EXEX",
    )

Prompt the user for a scalar or axis-wise boundary specification.

# Function description
For `dim == 1`, this helper behaves like a simple scalar string prompt. For
`dim > 1`, it accepts either one shared boundary value or exactly `dim`
comma-separated per-axis values.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `dim::Int`: Problem dimensionality.

# Keyword arguments
- `default_scalar::AbstractString = "LU_EXEX"`:
  Default shared boundary value.

# Returns
- `String` for shared scalar input.
- `Vector{String}` for explicit axis-wise input.

# Errors
- Throws if `dim > 1` and the user supplies neither one value nor exactly
  `dim` comma-separated values.
- Propagates I/O errors from [`_prompt`](@ref).
"""
function _prompt_boundary_spec(
    msg::String,
    dim::Int;
    default_scalar::AbstractString = "LU_EXEX",
)
    if dim == 1
        return _prompt(msg, default_scalar)
    end

    raw = _prompt(
        msg * " (single value or comma-separated $dim values)",
        default_scalar,
    )

    parts = String.(strip.(split(raw, ",")))

    if length(parts) == 1
        return parts[1]
    elseif length(parts) == dim
        return parts
    else
        error(
            "Boundary input must be either a single value or exactly $dim values, " *
            "but got $(length(parts)) value(s)."
        )
    end
end

"""
    _prompt_rule_spec(
        msg::String,
        dim::Int;
        default_scalar::AbstractString = "gauss_p4",
    )

Prompt the user for a scalar or axis-wise quadrature-rule specification.

# Function description
For `dim == 1`, this helper behaves like a simple scalar string prompt. For
`dim > 1`, it accepts either one shared rule value or exactly `dim`
comma-separated per-axis values.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `dim::Int`: Problem dimensionality.

# Keyword arguments
- `default_scalar::AbstractString = "gauss_p4"`:
  Default shared rule value.

# Returns
- `String` for shared scalar input.
- `Vector{String}` for explicit axis-wise input.

# Errors
- Throws if `dim > 1` and the user supplies neither one value nor exactly
  `dim` comma-separated values.
- Propagates I/O errors from [`_prompt`](@ref).
"""
function _prompt_rule_spec(
    msg::String,
    dim::Int;
    default_scalar::AbstractString = "gauss_p4",
)
    if dim == 1
        return _prompt(msg, default_scalar)
    end

    raw = _prompt(
        msg * " (single value or comma-separated $dim values)",
        default_scalar,
    )

    parts = String.(strip.(split(raw, ",")))

    if length(parts) == 1
        return parts[1]
    elseif length(parts) == dim
        return parts
    else
        error(
            "Rule input must be either a single value or exactly $dim values, " *
            "but got $(length(parts)) value(s)."
        )
    end
end

"""
    _toml_literal(x) -> String

Format a scalar or array-like wizard value as a TOML literal.

# Function description
This helper converts wizard values into the textual form embedded in the
generated TOML template.

Behavior depends on `x`:

- `AbstractString` values are emitted with `repr(x)`, so they become quoted TOML
  strings;
- tuple- and vector-like values are emitted recursively as bracketed arrays;
- non-string scalar values are emitted via `string(x)`.

# Arguments
- `x`: Scalar or array-like value to embed in the TOML output.

# Returns
- `String`: TOML literal representation of `x`.

# Errors
- No explicit validation is performed.

# Notes
- This helper is used so the wizard can preserve precision-sensitive domain
  input by writing it as TOML strings instead of eagerly parsing it as
  `Float64`.
"""
@inline function _toml_literal(x)
    if x isa AbstractString
        return repr(x)
    elseif x isa Tuple || x isa AbstractVector
        return "[" * join(_toml_literal.(collect(x)), ", ") * "]"
    else
        return string(x)
    end
end

"""
    _build_toml(cfg) -> String

Construct a wizard-generated [`TOML`](https://toml.io/en/) configuration string.

# Function description
This helper interpolates the fields of `cfg` into a fixed [`TOML`](https://toml.io/en/) template
with a predictable section order.

The generated sections are:

- `[integrand]`
- `[domain]`
- `[sampling]`
- `[quadrature]`
- `[error]`
- `[execution]`
- `[output]`

The current template supports scalar or array-valued domain endpoints.
When domain bounds are supplied as strings, they are emitted as quoted TOML
strings so precision-sensitive numeric text can be preserved until
[`Maranatha.Utils.MaranathaTOML.parse_run_config_from_toml`](@ref) interprets them using
`real_type`.

# Arguments
- `cfg`: Configuration bundle providing the fields required by the wizard TOML
  template.

# Returns
- `String`: [`TOML`](https://toml.io/en/) representation of the configuration.

# Errors
- Missing-field errors are propagated from property access on `cfg`.

# Notes
- The [`TOML`](https://toml.io/en/) string is assembled manually to preserve predictable ordering and
  readability.
- This helper does not validate or normalize the supplied values.
- String-valued domain bounds are intentionally preserved rather than eagerly
  parsed in the wizard.
"""
function _build_toml(cfg)

    a_toml = _toml_literal(cfg.a)
    b_toml = _toml_literal(cfg.b)
    ns_toml = _toml_literal(cfg.nsamples)

    return """
[integrand]
file = \"$(cfg.file)\"
name = \"$(cfg.func)\"

[domain]
a = $a_toml
b = $b_toml
dim = $(cfg.dim)

[sampling]
nsamples = $ns_toml

[quadrature]
rule = $(_toml_literal(cfg.rule))
boundary = $(_toml_literal(cfg.boundary))

[error]
err_method = \"$(cfg.err_method)\"
fit_terms = $(cfg.fit_terms)
nerr_terms = $(cfg.nerr_terms)
ff_shift = $(cfg.ff_shift)

[execution]
use_error_jet = $(cfg.use_error_jet)
use_cuda = $(cfg.use_cuda)
real_type = \"$(cfg.real_type)\"

[output]
name_prefix = \"$(cfg.name_prefix)\"
save_path = \"$(cfg.save_path)\"
write_summary = $(cfg.write_summary)
save_file = $(cfg.save_file)
"""
end

"""
    _build_sample_integrand(
        dim::Int;
        func_name::AbstractString = "integrand",
    ) -> String

Generate a sample integrand source-code string for the wizard.

# Function description
This helper constructs a minimal Julia function definition whose signature
matches the requested dimensionality `dim`.

For low dimensions, it emits simple analytic examples. For higher dimensions,
it emits a placeholder constant-valued function.

# Arguments
- `dim::Int`: Requested integrand dimensionality.

# Keyword arguments
- `func_name::AbstractString`: Name of the generated function.

# Returns
- `String`: Source-code string containing the sample integrand definition.

# Errors
- No explicit validation is performed.

# Notes
- The caller is responsible for ensuring that `dim >= 1`.
"""
function _build_sample_integrand(
    dim::Int;
    func_name::AbstractString = "integrand",
)::String
    if dim == 1
        return """
$(func_name)(x) = sin(x)
"""
    elseif dim == 2
        return """
$(func_name)(x, y) = sin(x * y)
"""
    elseif dim == 3
        return """
$(func_name)(x, y, z) = sin(x * y * z)
"""
    elseif dim == 4
        return """
$(func_name)(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
"""
    else
        args = join(["x$i" for i in 1:dim], ", ")
        return """
$(func_name)($(args)) = 0.0
"""
    end
end

"""
    _write_sample_integrand(
        path::AbstractString,
        dim::Int;
        func_name::AbstractString = "integrand",
    ) -> Nothing

Write a sample integrand source file to disk.

# Function description
This helper generates a sample integrand via
[`_build_sample_integrand`](@ref) and writes the resulting source code to
`path`.

# Arguments
- `path::AbstractString`: Output file path.
- `dim::Int`: Requested integrand dimensionality.

# Keyword arguments
- `func_name::AbstractString`: Name of the generated function.

# Returns
- `Nothing`.

# Errors
- Propagates file-writing errors from `open` / `write`.

# Notes
- Existing files are overwritten by the underlying write operation.
"""
function _write_sample_integrand(
    path::AbstractString,
    dim::Int;
    func_name::AbstractString = "integrand",
)::Nothing
    code = _build_sample_integrand(dim; func_name=func_name)

    open(path, "w") do io
        write(io, code)
    end

    return nothing
end

"""
    run_wizard(; 
        output_path::AbstractString = "maranatha.toml"
    ) -> Nothing

Launch the interactive Maranatha [`TOML`](https://toml.io/en/) configuration wizard.

# Function description
This routine interactively collects configuration values from the user,
constructs a [`TOML`](https://toml.io/en/) configuration string with the wizard's fixed
template, writes it to `output_path`, and optionally writes a sample integrand
source file.

The wizard currently prompts for:

- dimensionality `dim`,
- integrand file and function name,
- either scalar domain bounds or per-axis domain-bound arrays, depending on the
  domain-mode branch,
- sampling schedule,
- quadrature settings,
- error-estimation settings,
- execution flags,
- output configuration.

String-valued options such as `rule`, `boundary`, `err_method`, and
`real_type` are collected as raw user input and written directly into the TOML
file without local validation.

Domain bounds are collected as strings so precision-sensitive numeric text can
be preserved in the generated [`TOML`](https://toml.io/en/) file and later parsed using the
selected `real_type`.

# Keyword arguments
- `output_path::AbstractString`: Destination path of the generated [`TOML`](https://toml.io/en/) file.

# Returns
- `Nothing`.

# Errors
- Propagates input-parsing errors from the `_prompt_*` helpers.
- Propagates I/O errors from interactive input.
- Propagates file-writing errors when writing the [`TOML`](https://toml.io/en/) or sample integrand file.

# Notes
- If the selected sample integrand file already exists, the wizard asks whether
  it should be overwritten.
- In the per-axis domain branch, the wizard requires exactly `dim` lower-bound
  values and `dim` upper-bound values.
- For `dim > 1`, both `rule` and `boundary` may be entered either as one
  shared scalar value or as exactly `dim` comma-separated axis-wise values.
- The `use_error_jet` option only affects derivative-based error methods and is
  ignored when `err_method = "refinement"`.
"""
function run_wizard(;
    output_path::AbstractString = "maranatha.toml"
)

    println()
    println("Maranatha TOML configuration wizard")
    println("-----------------------------------")

    dim = _prompt_int("Dimension", 1)
    dim >= 1 || error("Dimension must be >= 1 (got dim=$dim).")

    file = _prompt("Integrand file", "sample_$(dim)d.jl")
    func = _prompt("Integrand function name", "integrand")

    use_axiswise_domain = dim > 1 ? _prompt_bool("Use axis-wise rectangular domain", false) : false

    if use_axiswise_domain
        println("Rectangular domain mode: enter comma-separated bounds with exactly $dim values.")

        a = _prompt_literal_vector("Lower bounds (a)", fill("0.0", dim))
        b = _prompt_literal_vector("Upper bounds (b)", fill("3.1415926535897932384626433832795028841971", dim))

        length(a) == dim || error(
            "Expected $dim lower-bound values for domain.a, got $(length(a))."
        )
        length(b) == dim || error(
            "Expected $dim upper-bound values for domain.b, got $(length(b))."
        )
    else
        a = _prompt("Lower bound (a)", "0.0")
        b = _prompt("Upper bound (b)", "3.1415926535897932384626433832795028841971")
    end

    nsamples = _prompt_int_vector("Number of samples", [2, 3, 4, 5, 6, 7, 8, 9])

    if dim > 1
        println("Rule can be either a single value shared by all axes,")
        println("or a comma-separated list with exactly $dim values.")
    end

    println("Example rules: newton_p3, gauss_p4, bspline_interp_p3, bspline_smooth_p3 ...")
    rule = _prompt_rule_spec("Quadrature rule", dim; default_scalar = "gauss_p4")

    println("Example boundaries: LU_ININ, LU_EXEX, LU_INEX, LU_EXIN ...")
    boundary = _prompt_boundary_spec("Boundary", dim; default_scalar = "LU_EXEX")

    println("Error methods: refinement, forwarddiff, taylorseries, enzyme, fastdifferentiation")
    err_method = _prompt("Error method", "refinement")

    if lowercase(err_method) == "refinement"
        println("Note: for err_method=refinement, nerr_terms is effectively treated as 0 internally.")
    end

    fit_terms = _prompt_int("fit_terms", 4)
    nerr_terms = _prompt_int("nerr_terms", 3)
    ff_shift = _prompt_int("ff_shift", 0)

    use_error_jet = _prompt_bool("Use error jet", false)

    use_cuda = _prompt_bool("Use CUDA", false)
    real_type = _prompt("Real type (Float32, Float64, Double64, BigFloat)", "Float64")

    name_prefix = _prompt("Output name prefix", "Maranatha")
    save_path = _prompt("Save path", ".")

    write_summary = _prompt_bool("Write summary", true)
    save_file = _prompt_bool("Save result file", true)

    make_sample_integrand = _prompt_bool("Generate sample integrand file", true)

    cfg = (
        file = file,
        func = func,
        a = a,
        b = b,
        dim = dim,
        nsamples = nsamples,
        rule = rule,
        boundary = boundary,
        err_method = err_method,
        fit_terms = fit_terms,
        nerr_terms = nerr_terms,
        ff_shift = ff_shift,
        use_error_jet = use_error_jet,
        use_cuda = use_cuda,
        real_type = real_type,
        name_prefix = name_prefix,
        save_path = save_path,
        write_summary = write_summary,
        save_file = save_file,
    )

    toml = _build_toml(cfg)

    open(output_path, "w") do io
        write(io, toml)
    end

    println()
    println("TOML file written to: $output_path")

    if make_sample_integrand
        write_integrand = true

        if isfile(file)
            write_integrand = _prompt_bool(
                "Integrand file already exists. Overwrite",
                false,
            )
        end

        if write_integrand
            _write_sample_integrand(file, dim; func_name = func)
            println("Sample integrand file written to: $file")
        else
            println("Skipped writing sample integrand file.")
        end
    end

    println()

    return nothing
end

end  # module Wizard
