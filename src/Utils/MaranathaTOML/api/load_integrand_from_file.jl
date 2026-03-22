# ============================================================================
# src/Utils/MaranathaTOML/api/load_integrand_from_file.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
