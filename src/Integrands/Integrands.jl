# ============================================================================
# src/Integrands/Integrands.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Integrands

Registry-based integrand factory utilities for `Maranatha.jl`.

# Module description
`Maranatha.Integrands` provides a small symbolic registry that maps integrand
names to factory functions. This lets higher-level workflows reconstruct
callable test or sample integrands from symbolic names plus keyword arguments.

# Main entry points
- [`register_integrand!`](@ref)
- [`integrand`](@ref)
- [`available_integrands`](@ref)

# Notes
- The registry is lightweight and intentionally does not impose a particular
  callable type beyond the factory convention.
- Re-registering an existing name overwrites the stored factory.
"""
module Integrands

import ..Utils.JobLoggerTools

# ============================================================
# Registry: name => factory
#
# Factory signature:
#     factory(; kwargs...) -> callable integrand
# ============================================================

"""
    INTEGRAND_REGISTRY :: Dict{Symbol, Function}

Module-level registry mapping integrand names to factory functions.

# Description
`INTEGRAND_REGISTRY` stores the named integrand factories used by
[`register_integrand!`](@ref), [`integrand`](@ref), and [`available_integrands`](@ref).

Each entry has the form:

- key   : `Symbol` integrand name
- value : `Function` factory

where the factory is expected to follow the convention:

    factory(; kwargs...) -> callable_integrand

That is, the stored function should accept keyword arguments and return a
callable integrand object, such as a closure or a callable struct.

# Purpose
This registry provides a lightweight indirection layer so that integrands can be:

- registered once under a symbolic name,
- reconstructed later with runtime parameters,
- discovered through registry inspection.

# Notes
- Re-registering the same name overwrites the previous factory.
- The dictionary is mutable, but the binding itself is constant.
- Iteration order is not guaranteed to be stable and should not be relied on.
"""
const INTEGRAND_REGISTRY = Dict{Symbol, Function}()

"""
    register_integrand!(
        name::Symbol, 
        factory::Function
    )

Register an integrand factory in the module-level registry.

# Function description
This function associates the symbolic name `name` with a factory callable
`factory`. The factory is expected to accept keyword arguments and return a
callable integrand object.

Once registered, the integrand can later be constructed through
[`integrand`](@ref)`(name; kwargs...)`.

# Arguments
- `name::Symbol`: Registry key used to identify the integrand.
- `factory::Function`: Factory callable of the form `factory(; kwargs...) -> f`,
  where `f` is itself callable.

# Returns
- `nothing`

# Errors
- No explicit validation is performed; re-registration overwrites any existing
  entry under the same name.

# Notes
- The registry is stored in [`INTEGRAND_REGISTRY`](@ref).
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
This function looks up `name` in [`INTEGRAND_REGISTRY`](@ref), retrieves the
registered factory, and invokes it with the provided keyword arguments.

The returned object is expected to be directly callable and usable by the
higher-level Maranatha workflow.

# Arguments
- `name::Symbol`: Integrand identifier registered via [`register_integrand!`](@ref).

# Keyword arguments
- `kwargs...`: Keyword arguments forwarded to the registered factory.

# Returns
- A callable integrand object returned by the factory associated with `name`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `name` is not registered.

# Notes
- The error message includes currently available registry keys.
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

# Function description
This helper exposes the current registry keys as a `Vector{Symbol}`.

# Arguments
- None.

# Returns
- `Vector{Symbol}`: Registered integrand names.

# Errors
- No explicit errors are thrown.

# Notes
- The returned order follows the iteration order of the internal dictionary and
  should not be treated as a stable sorted order.
"""
available_integrands() = collect(keys(INTEGRAND_REGISTRY))

end # module Integrands