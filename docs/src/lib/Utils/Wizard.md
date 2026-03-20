# Maranatha.Utils.Wizard

## Overview

`Maranatha.Utils.Wizard` provides the interactive command-line wizard used to
generate Maranatha [`TOML`](https://toml.io/en/) configuration files.

Its purpose is to lower the barrier for creating usable starting configurations
by walking the user through a sequence of prompts rather than requiring them to
author the [`TOML`](https://toml.io/en/) file manually from scratch.

The wizard is intentionally lightweight: it collects and writes values, but it
does not perform full semantic validation of the generated configuration.

It can now preserve precision-sensitive domain input by writing domain bounds
as TOML strings, leaving final numeric interpretation to the TOML parsing
layer.

---

## Input helper layer

The helpers

- [`Maranatha.Utils.Wizard._prompt`](@ref)
- [`Maranatha.Utils.Wizard._prompt_float`](@ref)
- [`Maranatha.Utils.Wizard._prompt_int`](@ref)
- [`Maranatha.Utils.Wizard._prompt_bool`](@ref)
- [`Maranatha.Utils.Wizard._prompt_int_vector`](@ref)
- [`Maranatha.Utils.Wizard._prompt_literal_vector`](@ref)

form the low-level interactive input layer.

These routines keep the main wizard logic simpler by separating:

- plain string prompting,
- numeric parsing,
- Boolean interpretation,
- comma-separated integer-vector parsing,
- precision-preserving literal collection for domain input.

The design is intentionally lightweight and suitable for trusted interactive
CLI use rather than fully defensive form validation.

---

## [`TOML`](https://toml.io/en/) construction

The helper [`Maranatha.Utils.Wizard._build_toml`](@ref) converts the collected configuration bundle
into a [`TOML`](https://toml.io/en/) string with a fixed, predictable section order.

The emitted template currently reflects the wizard's present prompt flow:
scalar or array-valued `[domain].a` / `[domain].b`, a flat `[sampling].nsamples`
array, and string-valued selectors such as `rule`, `boundary`, `err_method`,
and `real_type` written directly from user input.

When domain bounds are supplied as strings, the wizard emits them as quoted TOML
strings so precision-sensitive numeric text can be preserved until the parser
later interprets them using `real_type`.

The manual assembly is intentional: it keeps the output readable and stable,
which is useful when users later inspect or edit the generated file.

---

## Sample integrand generation

The helpers

- [`Maranatha.Utils.Wizard._build_sample_integrand`](@ref)
- [`Maranatha.Utils.Wizard._write_sample_integrand`](@ref)

provide a small convenience feature of the wizard: optionally generating a
sample Julia integrand file that matches the requested dimensionality.

This gives the user a runnable starting point immediately after the [`TOML`](https://toml.io/en/) file is
created.

---

## Main workflow

[`Maranatha.Utils.Wizard.run_wizard`](@ref) ties the module together.

Its workflow is:

1. prompt the user for configuration values,
2. assemble a configuration bundle,
3. build and write the [`TOML`](https://toml.io/en/) file,
4. optionally write a sample integrand file.

The current prompt flow gathers dimension, integrand metadata, either scalar
domain bounds or per-axis domain-bound arrays, sample counts, quadrature
labels, error settings, execution flags, and output options.

Option-like strings are written directly into the generated [`TOML`](https://toml.io/en/) file
without local validation. Domain bounds may also be preserved as strings so the
later [`TOML`](https://toml.io/en/) parser can interpret them in the precision selected by `real_type`.

This makes the wizard a practical front-end for first-time or quick setup
scenarios.

---

## Design scope

`Maranatha.Utils.Wizard` is an interactive setup helper.

It does **not**:

It does **not**:

- validate the full semantics of the generated configuration,
- execute the run itself,
- analyze the integrand mathematically,
- replace the [`TOML`](https://toml.io/en/) parsing / validation layer.

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