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

Prompt the user for a string value, with a default fallback.

# Function description

This helper prints an interactive prompt of the form

    msg [default]:

and reads a single line from standard input.

If the user enters an empty string, the stripped default value is returned.
Otherwise, the stripped user input is returned.

# Arguments

`msg::String`
: Prompt message shown to the user.

`default::AbstractString`
: Default value used when the user presses Enter without typing anything.

# Returns

A stripped `String` containing either the user input or the default value.

# Errors

This function does not perform validation beyond basic input reading.
Any I/O errors from `readline()` are propagated.

# Notes

This is a low-level helper used by the TOML wizard input routines.
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

This helper calls [`_prompt`](@ref) to obtain a string input and then parses
the result as `Float64`.

If the user presses Enter without typing anything, the provided default value
is used.

# Arguments

`msg::String`
: Prompt message shown to the user.

`default::Float64`
: Default floating-point value.

# Returns

A `Float64` parsed from the entered or default string.

# Errors

Throws an error if the entered value cannot be parsed as `Float64`.

# Notes

This helper is intended for TOML fields such as `a` and `b`.
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

This helper calls [`_prompt`](@ref) to obtain a string input and then parses
the result as `Int`.

If the user presses Enter without typing anything, the provided default value
is used.

# Arguments

`msg::String`
: Prompt message shown to the user.

`default::Int`
: Default integer value.

# Returns

An `Int` parsed from the entered or default string.

# Errors

Throws an error if the entered value cannot be parsed as `Int`.

# Notes

This helper is intended for TOML fields such as `dim`, `fit_terms`,
`nerr_terms`, and `ff_shift`.
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

This helper calls [`_prompt`](@ref) to obtain a string input and interprets the
result as a Boolean.

The following case-insensitive inputs are treated as `true`:

* `"true"`
* `"t"`
* `"1"`
* `"yes"`
* `"y"`

All other inputs are interpreted as `false`.

If the user presses Enter without typing anything, the provided default value
is used.

# Arguments

`msg::String`
: Prompt message shown to the user.

`default::Bool`
: Default Boolean value.

# Returns

A `Bool` determined from the entered or default string.

# Errors

This helper does not throw parsing errors for unrecognized inputs; instead,
any input not matching the accepted true-values is treated as `false`.

# Notes

This permissive behavior is suitable for a simple interactive wizard, but it is
less strict than a full validation parser.
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

This helper calls [`_prompt`](@ref) to obtain a string input, splits the result
by commas, and parses each entry as `Int`.

If the user presses Enter without typing anything, the provided default vector
is converted into a comma-separated string and reused.

# Arguments

`msg::String`
: Prompt message shown to the user.

`default::Vector{Int}`
: Default integer vector.

# Returns

A `Vector{Int}` parsed from the entered or default comma-separated string.

# Errors

Throws an error if any element cannot be parsed as `Int`.

# Notes

This helper is primarily intended for the `[sampling].nsamples` field in the
generated TOML configuration.
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

Construct a Maranatha TOML configuration string from a configuration bundle.

# Function description

This helper converts a configuration container `cfg` into a TOML-formatted
string with a fixed output order.

The generated sections are:

* `[integrand]`
* `[domain]`
* `[sampling]`
* `[quadrature]`
* `[error]`
* `[execution]`
* `[output]`

# Arguments

`cfg`
: Configuration object providing fields such as
  `file`, `func`, `a`, `b`, `dim`, `nsamples`, `rule`, `boundary`,
  `err_method`, `fit_terms`, `nerr_terms`, `ff_shift`, `use_threads`,
  `name_prefix`, `save_path`, `write_summary`, and `save_file`.

# Returns

A `String` containing the TOML representation of the configuration.

# Errors

This function assumes that `cfg` provides all required fields.
Missing-field errors are propagated from property access.

# Notes

The TOML string is assembled manually in order to preserve a predictable and
readable section order.
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
use_threads = $(cfg.use_threads)

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
        func_name::AbstractString="integrand",
    ) -> String

Generate a simple **sample integrand source code string** for the Maranatha wizard.

# Function description
This helper constructs a minimal Julia function definition suitable for use
as an example integrand file in a Maranatha configuration workflow.

The generated function signature depends on the requested dimensionality `dim`.
For common low dimensions (`1-4`), a simple analytic expression is used.
For higher dimensions, a placeholder constant function is produced.

The returned string is intended to be written directly to a `.jl` file.

# Arguments
- `dim::Int`  
  Number of integration dimensions.

# Keyword arguments
- `func_name::AbstractString="integrand"`  
  Name of the generated integrand function.

# Returns
- `String`  
  Source code string containing a Julia function definition.

# Errors
No explicit validation is performed here.  
The caller is responsible for ensuring that `dim ≥ 1`.
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
    _build_sample_integrand(
        dim::Int;
        func_name::AbstractString="integrand",
    ) -> String

Generate a simple **sample integrand source code string** for the Maranatha wizard.

# Function description
This helper constructs a minimal Julia function definition suitable for use
as an example integrand file in a Maranatha configuration workflow.

The generated function signature depends on the requested dimensionality `dim`.
For common low dimensions (`1-4`), a simple analytic expression is used.
For higher dimensions, a placeholder constant function is produced.

The returned string is intended to be written directly to a `.jl` file.

# Arguments
- `dim::Int`  
  Number of integration dimensions.

# Keyword arguments
- `func_name::AbstractString="integrand"`  
  Name of the generated integrand function.

# Returns
- `String`  
  Source code string containing a Julia function definition.

# Errors
No explicit validation is performed here.  
The caller is responsible for ensuring that `dim ≥ 1`.
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
    run_wizard(; output_path::AbstractString="maranatha.toml") -> Nothing

Interactive wizard for generating a **Maranatha TOML configuration file**.

# Function description
This routine launches an interactive command-line wizard that asks the user
a sequence of questions and constructs a valid Maranatha configuration file.

The wizard collects information about:

- integrand location and function name
- integration domain and dimensionality
- sampling schedule
- quadrature rule and boundary condition
- error estimation settings
- execution options
- output configuration

The resulting configuration is written to a TOML file that can be passed
directly to the Maranatha runner.

Optionally, the wizard can also generate a **sample integrand `.jl` file**
matching the selected dimensionality.

# Keyword arguments
- `output_path::AbstractString="maranatha.toml"`  
  Destination path for the generated TOML configuration file.

# Returns
- `Nothing`

# Errors
Input parsing errors from the `_prompt_*` helpers will propagate if
invalid user input is provided.
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

    println("Error methods: forwarddiff, taylorseries, enzyme, fastdifferentiation")
    err_method = _prompt("Error method", "forwarddiff")

    fit_terms = _prompt_int("fit_terms", 4)
    nerr_terms = _prompt_int("nerr_terms", 3)
    ff_shift = _prompt_int("ff_shift", 0)

    use_threads = _prompt_bool("Use threads", true)

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
        use_threads = use_threads,
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