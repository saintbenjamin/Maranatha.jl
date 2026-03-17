# Maranatha.Utils.MaranathaTOML

## Overview

`Maranatha.Utils.MaranathaTOML` provides the [`TOML`](https://toml.io/en/)-facing configuration layer for
`Maranatha.jl`.

Its purpose is to translate user-facing [`TOML`](https://toml.io/en/) configuration files into the
normalized in-memory configuration structures expected by the run pipeline, and
to reject obviously invalid configurations before execution begins.

---

## Core responsibilities

This module mainly handles three tasks:

- defining the supported `err_method` identifiers,
- loading user integrands from external Julia files,
- parsing and validating [`TOML`](https://toml.io/en/)-based run configurations.

Together, these helpers form the configuration boundary between user-authored
run files and the rest of the Maranatha execution stack.

---

## Supported error methods

The constant [`Maranatha.Utils.MaranathaTOML.VALID_ERR_METHODS`](@ref) defines the currently accepted
error-estimation backend identifiers.

This gives the [`TOML`](https://toml.io/en/)-validation layer a single authoritative list of supported
values instead of duplicating that knowledge across parsing and execution code.

---

## User integrand loading

The helper [`Maranatha.Utils.MaranathaTOML.load_integrand_from_file`](@ref) evaluates a user-supplied Julia
source file inside a freshly created module and extracts a named function from
it.

The use of an isolated module is important because it prevents user-defined
symbols from leaking into the main package namespace.

This mechanism is convenient, but it is also intentionally simple: it assumes
trusted local Julia files rather than sandboxed external code.

---

## Parse versus validate

The module separates configuration handling into two stages:

### [`Maranatha.Utils.MaranathaTOML.parse_run_config_from_toml`](@ref)

Responsible for:

- reading the [`TOML`](https://toml.io/en/) file,
- extracting known sections,
- resolving relative paths,
- normalizing values into a predictable `NamedTuple`.

### [`Maranatha.Utils.MaranathaTOML.validate_run_config`](@ref)

Responsible for:

- structural checks,
- simple numerical checks,
- supported-option checks,
- existence checks for referenced files.

This separation keeps the parsing stage focused on representation conversion,
while the validation stage focuses on execution readiness.

---

## Path normalization policy

A key design detail is that relative paths in the [`TOML`](https://toml.io/en/) file are interpreted
relative to the [`TOML`](https://toml.io/en/) file's own directory, not the current working directory.

This makes configuration files more portable and self-contained, since their
associated integrand paths and save paths move naturally with the config file.

---

## Design scope

`Maranatha.Utils.MaranathaTOML` is a configuration module.

It does **not**:

- execute quadrature,
- evaluate the integrand,
- verify full integrand/dimension compatibility,
- replace the numerical execution layer.

Instead, it prepares and sanity-checks the configuration data that later stages
will use.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.MaranathaTOML,
]
Private = true
```