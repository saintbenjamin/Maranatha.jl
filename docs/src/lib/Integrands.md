# Maranatha.Integrands

## Overview

`Maranatha.Integrands` provides a lightweight registry layer for named integrand
factories.

Its purpose is to let the package refer to integrands by symbolic name while
still constructing concrete callable objects only when needed.

---

## Registry design

The registry is stored as

```julia
const INTEGRAND_REGISTRY = Dict{Symbol, Function}()
```

Each entry maps:

- a symbolic integrand name,
- to a factory function that accepts keyword arguments and returns a callable
  integrand.

So the intended pattern is:

```julia
register_integrand!(:my_integrand, (; kwargs...) -> f)
```

followed by

```julia
f = integrand(:my_integrand; kwargs...)
```

---

## Why factories instead of raw functions

The registry stores **factories**, not just already-constructed callables.

That allows each named integrand to be parameterized at construction time,
which is useful when an integrand depends on user-selected constants,
shifts, masses, frequencies, amplitudes, or other runtime options.

This design also works naturally with:

- ordinary functions,
- closures,
- callable structs.

---

## Public helpers

### [`Maranatha.Integrands.register_integrand!`](@ref)

Adds or replaces a registry entry.

### [`Maranatha.Integrands.integrand`](@ref)

Constructs a callable integrand from a registered name.

### [`Maranatha.Integrands.available_integrands`](@ref)

Returns the currently registered names.

---

## Design scope

This module is intentionally minimal.

It does **not**:

- evaluate integrals,
- perform quadrature,
- validate mathematical properties of registered functions,
- enforce a specific callable type beyond practical compatibility.

Its responsibility is simply to provide a small naming-and-construction layer
for integrands used elsewhere in `Maranatha.jl`.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Integrands,
]
Private = true
```