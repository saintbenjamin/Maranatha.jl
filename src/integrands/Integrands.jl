# ============================================================================
# src/integrands/Integrands.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Integrands

using ..JobLoggerTools

export integrand, register_integrand!, available_integrands

# ============================================================
# Registry: name => factory
#
# Factory signature:
#     factory(; kwargs...) -> callable integrand
# ============================================================

const INTEGRAND_REGISTRY = Dict{Symbol, Function}()

"""
    register_integrand!(
        name::Symbol, 
        factory::Function
    )

Register a new integrand factory into the `Maranatha.jl` integrand registry.

# Function description
This function associates an integrand name `name` with a factory function
`factory`. The factory must accept keyword arguments and return a callable
integrand object (e.g., a closure or a callable struct).

Once registered, the integrand can be constructed via:

`integrand(name; kwargs...)`.

# Arguments
- `name::Symbol`: Integrand identifier used as the registry key.
- `factory::Function`: Factory function of the form `factory(; kwargs...) -> f`,
  where `f` is callable.

# Returns
- `nothing`

# Notes
- Re-registering an existing `name` overwrites the prior factory.
- The registry is stored as a module-level constant dictionary to keep
  lookup/dispatch lightweight.
"""
function register_integrand!(
    name::Symbol, 
    factory::Function
)
    INTEGRAND_REGISTRY[name] = factory
    return nothing
end

"""
    integrand(
        name::Symbol; 
        kwargs...
    )

Construct a callable integrand from the registry.

# Function description
This function looks up `name` in the integrand registry and invokes the
corresponding factory with the provided keyword arguments. The result is a
callable object that can be passed directly into [`Maranatha.Runner.run_Maranatha`](@ref).

# Arguments
- `name::Symbol`: Integrand identifier registered via [`register_integrand!`](@ref).

# Keyword arguments
- `kwargs...`: Keyword arguments forwarded to the registered factory.

# Returns
- A callable integrand object returned by the registered factory.

# Errors
- Throws an error if `name` is not registered.
"""
function integrand(
    name::Symbol; 
    kwargs...
)
    haskey(INTEGRAND_REGISTRY, name) ||
        JobLoggerTools.error_benji("Unknown integrand: $(name). Available: $(collect(keys(INTEGRAND_REGISTRY)))")

    factory = INTEGRAND_REGISTRY[name]
    return factory(; kwargs...)
end

"""
    available_integrands()

Return the list of currently registered integrand names.

# Returns
- `Vector{Symbol}`: Registered integrand keys.

# Notes
- The order of the returned symbols follows the iteration order of the internal
  dictionary and is not guaranteed to be stable across Julia versions.
"""
available_integrands() = collect(keys(INTEGRAND_REGISTRY))

end # module Integrands