# Maranatha.Utils.Wizard

## Overview

`Maranatha.Utils.Wizard` provides the interactive command-line wizard used to
generate Maranatha `TOML` configuration files.

Its purpose is to lower the barrier for creating valid run configurations by
walking the user through a sequence of prompts rather than requiring them to
author the `TOML` file manually from scratch.

---

## Input helper layer

The helpers

- [`Maranatha.Utils.Wizard._prompt`](@ref)
- [`Maranatha.Utils.Wizard._prompt_float`](@ref)
- [`Maranatha.Utils.Wizard._prompt_int`](@ref)
- [`Maranatha.Utils.Wizard._prompt_bool`](@ref)
- [`Maranatha.Utils.Wizard._prompt_int_vector`](@ref)

form the low-level interactive input layer.

These routines keep the main wizard logic simpler by separating:

- plain string prompting,
- numeric parsing,
- Boolean interpretation,
- comma-separated integer-vector parsing.

The design is intentionally lightweight and suitable for trusted interactive
CLI use rather than fully defensive form validation.

---

## `TOML` construction

The helper [`Maranatha.Utils.Wizard._build_toml`](@ref) converts the collected configuration bundle
into a `TOML` string with a fixed, predictable section order.

The manual assembly is intentional: it keeps the output readable and stable,
which is useful when users later inspect or edit the generated file.

---

## Sample integrand generation

The helpers

- [`Maranatha.Utils.Wizard._build_sample_integrand`](@ref)
- [`Maranatha.Utils.Wizard._write_sample_integrand`](@ref)

provide a small convenience feature of the wizard: optionally generating a
sample Julia integrand file that matches the requested dimensionality.

This gives the user a runnable starting point immediately after the `TOML` file is
created.

---

## Main workflow

[`Maranatha.Utils.Wizard.run_wizard`](@ref) ties the module together.

Its workflow is:

1. prompt the user for configuration values,
2. assemble a configuration bundle,
3. build and write the `TOML` file,
4. optionally write a sample integrand file.

This makes the wizard a practical front-end for first-time or quick setup
scenarios.

---

## Design scope

`Maranatha.Utils.Wizard` is an interactive setup helper.

It does **not**:

- validate the full semantics of the generated configuration,
- execute the run itself,
- analyze the integrand mathematically,
- replace the `TOML` parsing / validation layer.

Instead, it focuses on generating a usable starting configuration in a guided
way.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.Wizard,
]
Private = true
```