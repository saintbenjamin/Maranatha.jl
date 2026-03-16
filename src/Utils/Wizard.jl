# ============================================================================
# src/Utils/Wizard.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Wizard

"""
    _prompt(
        msg::String,
        default::AbstractString
    ) -> String

Prompt the user for a string value with a default fallback.

# Function description
This helper prints an interactive prompt of the form

    msg [default]:

reads one line from standard input, and returns either the stripped user input
or the stripped default value if the input is empty.

# Arguments
- `msg::String`: Prompt message shown to the user.
- `default::AbstractString`: Default value used when the user presses Enter.

# Returns
- `String`: Stripped user input or stripped default value.

# Errors
- Propagates I/O errors from `readline()`.

# Notes
- This is a low-level helper used by the wizard input routines.
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
- Intended for numeric [`TOML`](https://toml.io/en/) fields such as `a` and `b`.
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
    parse.(Int, split(s, ","))
end

"""
    _build_toml(cfg) -> String

Construct a [`TOML`](https://toml.io/en/) configuration string from a wizard configuration bundle.

# Function description
This helper converts a configuration container into a [`TOML`](https://toml.io/en/)-formatted string
with a fixed section order.

The generated sections are:

- `[integrand]`
- `[domain]`
- `[sampling]`
- `[quadrature]`
- `[error]`
- `[execution]`
- `[output]`

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
"""
function _build_toml(cfg)

    ns = "[" * join(cfg.nsamples, ", ") * "]"

    return """
[integrand]
file = \"$(cfg.file)\"
name = \"$(cfg.func)\"

[domain]
a = $(cfg.a)
b = $(cfg.b)
dim = $(cfg.dim)

[sampling]
nsamples = $ns

[quadrature]
rule = \"$(cfg.rule)\"
boundary = \"$(cfg.boundary)\"

[error]
err_method = \"$(cfg.err_method)\"
fit_terms = $(cfg.fit_terms)
nerr_terms = $(cfg.nerr_terms)
ff_shift = $(cfg.ff_shift)

[execution]
use_error_jet = $(cfg.use_error_jet)

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
constructs a [`TOML`](https://toml.io/en/) configuration string, writes it to disk, and optionally
writes a sample integrand source file.

The wizard gathers information about:

- integrand file and function name,
- integration domain and dimensionality,
- sampling schedule,
- quadrature settings,
- error-estimation settings,
- execution flags,
- output configuration.

# Keyword arguments
- `output_path::AbstractString`: Destination path of the generated [`TOML`](https://toml.io/en/) file.

# Returns
- `Nothing`.

# Errors
- Propagates input-parsing errors from the `_prompt_*` helpers.
- Propagates file-writing errors when writing the [`TOML`](https://toml.io/en/) or sample integrand file.

# Notes
- If the selected sample integrand file already exists, the wizard asks whether
  it should be overwritten.
- The `use_error_jet` option only affects derivative-based error methods and is
  ignored when `err_method = "refinement"`.
"""
function run_wizard(;
    output_path::AbstractString = "maranatha.toml"
)

    println()
    println("Maranatha TOML configuration wizard")
    println("-----------------------------------")

    file = _prompt("Integrand file", "sample_1d.jl")
    func = _prompt("Integrand function name", "integrand")

    a = _prompt_float("Lower bound (a)", 0.0)
    b = _prompt_float("Upper bound (b)", 3.141592653589793)
    dim = _prompt_int("Dimension", 1)

    nsamples = _prompt_int_vector("Number of samples", [2, 3, 4, 5, 6, 7, 8, 9])

    println("Example rules: newton_p3, gauss_p4, bspline_p3 ...")
    rule = _prompt("Quadrature rule", "gauss_p4")

    println("Example boundaries: LU_ININ, LU_EXEX, LU_INEX ...")
    boundary = _prompt("Boundary", "LU_EXEX")

    println("Error methods: refinement, forwarddiff, taylorseries, enzyme, fastdifferentiation")
    err_method = _prompt("Error method", "refinement")

    fit_terms = _prompt_int("fit_terms", 4)
    nerr_terms = _prompt_int("nerr_terms", 3)
    ff_shift = _prompt_int("ff_shift", 0)

    use_error_jet = _prompt_bool("Use error jet", false)

    name_prefix = _prompt("Output name prefix", "1D")
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