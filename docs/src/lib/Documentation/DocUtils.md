# Maranatha.Documentation.DocUtils

`Maranatha.Documentation.DocUtils` contains small reusable helpers shared by
the plotting and reporting layers of `Maranatha.jl`.

Unlike `PlotTools` and `Reporter`, this module does not generate figures or
reports by itself. Its role is to provide stable helper logic that would
otherwise be duplicated across those higher-level output modules.

---

## Overview

The current helper set focuses on two recurring documentation tasks:

- splitting a user-facing report name into a display label and a
  filesystem-safe basename,
- constructing compact filename tokens from scalar or axis-wise
  `rule` / `boundary` specifications.

This keeps figure names, report basenames, and related output artifacts
consistent across the documentation stack.

---

## Current responsibilities

| Helper | Responsibility |
|:--|:--|
| [`Maranatha.Documentation.DocUtils._split_report_name`](@ref) | separate a human-facing report name from a filesystem basename |
| [`Maranatha.Documentation.DocUtils._report_name_cfg_dim`](@ref) | infer the effective axis count implied by domain / rule / boundary metadata |
| [`Maranatha.Documentation.DocUtils._rule_boundary_filename_token`](@ref) | build the compact scalar-or-axis-wise rule / boundary token used in filenames |

---

## Filename-token policy

The helper
[`Maranatha.Documentation.DocUtils._rule_boundary_filename_token`](@ref)
follows the package-wide filename convention:

- scalar `rule` and scalar `boundary`:
  `rule_boundary`
- any axis-wise domain / rule / boundary metadata:
  `1_rule_1_boundary_1_2_rule_2_boundary_2_...`

The domain endpoints themselves are used only to infer the effective axis
count. They are not embedded directly in output filenames.

---

## Notes

- These helpers are internal, but they are intentionally documented because the
  documentation-output layer depends on them heavily.
- Axis-wise expansion is shared conceptually with the reporting helper module,
  but `DocUtils` is specifically concerned with filename and display-name
  normalization.

---

## API reference

```@autodocs
Modules = [
    Maranatha.Documentation.DocUtils,
]
Private = true
```
