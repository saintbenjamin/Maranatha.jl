# Maranatha.Utils.MaranathaTOML

## Overview

`Maranatha.Utils.MaranathaTOML` provides the [`TOML`](https://toml.io/en/)-facing configuration layer for
`Maranatha.jl`.

Its purpose is to translate user-facing [`TOML`](https://toml.io/en/) configuration files into the
normalized in-memory configuration structures expected by the run pipeline, and
to reject obviously invalid configurations before execution begins.

This now includes `real_type`-aware parsing of domain endpoints, so
`[domain].a` and `[domain].b` may be supplied either as ordinary numeric TOML
values or as string-valued numeric literals when precision should be preserved
until parse time.

---

## Core responsibilities

This module mainly handles three tasks:

- defining the supported `err_method` identifiers,
- loading user integrands from external Julia files,
- parsing and validating [`TOML`](https://toml.io/en/)-based run configurations, including
  `real_type`-aware domain parsing.

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

The helper [`Maranatha.Utils.MaranathaTOML.load_integrand_from_file`](@ref) resolves a user-supplied Julia
source file to an absolute path, evaluates it inside a freshly created module,
and extracts a named binding from it after verifying that the binding is a
function.

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
- resolving relative paths relative to the [`TOML`](https://toml.io/en/) file directory,
- converting selectors such as `rule`, `boundary`, `err_method`, and `real_type`
  into `Symbol` values,
- parsing `[domain].a` and `[domain].b` in the scalar type selected by `real_type`,
- accepting domain endpoints either as ordinary numeric TOML values or as
  string-valued numeric literals,
- normalizing collection-valued domain endpoints into vectors,
- returning a predictable `NamedTuple`,
- performing a small amount of structural checking before full validation.

### [`Maranatha.Utils.MaranathaTOML.validate_run_config`](@ref)

Responsible for:

- structural checks,
- dimensional and domain-bound checks,
- `nsamples` checks such as emptiness, positivity, and duplicate rejection,
- supported-option checks for `err_method` and `real_type`,
- CUDA compatibility checks for `real_type`,
- existence checks for referenced integrand files.

This separation keeps the parsing stage focused on representation conversion,
`real_type`-aware domain parsing, and light structural sanity checks, while the
validation stage focuses on execution readiness.

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