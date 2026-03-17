# Maranatha.Utils

## Overview

`Maranatha.Utils` is the shared utility aggregation layer of `Maranatha.jl`.

Its role is to collect cross-cutting helper submodules under one stable
namespace, rather than scattering logging, formatting, I/O, and workflow helpers
throughout the main numerical modules.

This module is intentionally infrastructural: it supports the rest of the
package, but it is not itself a numerical-method module.

---

## Included submodules

The current bundle includes the following utility submodules.

### [`Maranatha.Utils.JobLoggerTools`](@ref)

Provides logging and job-style status helpers, including timestamped printing,
stage delimiters, assertion-style helpers, and timing/reporting macros.

This is the utility layer most directly tied to runtime diagnostics and user
feedback during longer computations.

### [`Maranatha.Utils.AvgErrFormatter`](@ref)

Provides compact numerical formatting helpers for central values and
uncertainties.

This layer is useful when presenting fitted values, quadrature results, or
error estimates in parenthetical notation suitable for analysis logs and
publication-style output.

### [`Maranatha.Utils.MaranathaIO`](@ref)

Provides utility helpers related to package I/O behavior.

This submodule belongs here because file reading/writing is infrastructure
shared by multiple higher-level components.

### [`Maranatha.Utils.MaranathaTOML`](@ref)

Provides [`TOML`](https://toml.io/en/)-related helpers and configuration parsing support.

This is part of the utility layer because [`TOML`](https://toml.io/en/) configuration handling is a
package-wide concern rather than belonging to any one numerical subsystem.

### [`Maranatha.Utils.Wizard`](@ref)

Provides wizard-style helper utilities used for guided or structured setup
workflows.

This submodule also fits naturally into the shared utility namespace because it
supports package interaction and configuration rather than quadrature or fitting
logic directly.

---

## Why this aggregation layer exists

Without `Maranatha.Utils`, each higher-level module would need to import several
helper modules individually, and the codebase would become harder to navigate.

By collecting shared helpers under one layer, the package gains:

- a more stable namespace,
- clearer separation between infrastructure and algorithms,
- easier future extension,
- simpler discoverability for users and developers,
- a unified location for runtime support tools.

This aggregation includes components ranging from logging and configuration
handling to workflow helpers, all of which are
orthogonal to the scientific core of `Maranatha.jl`.

---

## Design scope

`Maranatha.Utils` is intentionally narrow in purpose.

It does **not**:

- define quadrature rules,
- estimate truncation errors,
- perform least-`\chi^2` fitting,
- define integrand mathematics.

Instead, it supports those systems by providing reusable infrastructure such as:

- logging,
- formatting,
- configuration handling,
- file I/O,
- setup helpers.

---

## Practical note

As the project evolves, new helper submodules can be added here without changing
the overall design philosophy: keep shared infrastructure modular, lightweight,
and clearly separated from the scientific core of `Maranatha.jl`.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils,
]
Private = true
```