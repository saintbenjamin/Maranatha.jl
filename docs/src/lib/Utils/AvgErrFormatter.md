# Maranatha.Utils.AvgErrFormatter

## Overview

`Maranatha.Utils.AvgErrFormatter` provides compact formatting utilities for
central values and uncertainties.

Its main purpose is to convert numeric `(average, error)` pairs into concise
parenthetical strings such as:

- `1.23(45)`
- `2.00000000000000(39)`
- `1.2(3) *`

which are convenient for logs, tables, and publication-style reporting.

---

## Main formatting workflow

The central formatting path is:

1. build or receive central-value and error strings,
2. attempt compact uncertainty formatting with [`Maranatha.Utils.AvgErrFormatter.avgerr_e2d`](@ref),
3. optionally apply visual digit grouping with
   [`Maranatha.Utils.AvgErrFormatter.latex_group_fraction_digits`](@ref).

The convenience wrapper [`Maranatha.Utils.AvgErrFormatter.avgerr_e2d_from_float`](@ref) starts from `Float64`
inputs and prepares the standard scientific-notation strings expected by the
main formatter.

---

## Failure-tolerant design

A major design feature of this module is that it tries hard **not** to fail
catastrophically during formatting.

If compact output construction becomes unreliable, the logic falls back to an
explicit scientific-notation string through
[`Maranatha.Utils.AvgErrFormatter._avgerr_fallback_string`](@ref).

This means formatting problems do not destroy the numerical content; they only
change how compactly it is displayed.

The helper [`Maranatha.Utils.AvgErrFormatter._safe_avgerr_return`](@ref) exists specifically to guard the final
dynamic `Printf` formatting stage.

---

## Digit grouping helpers

The helpers

- [`Maranatha.Utils.AvgErrFormatter._group_digits_right`](@ref)
- [`Maranatha.Utils.AvgErrFormatter._group_digits_left`](@ref)
- [`Maranatha.Utils.AvgErrFormatter.latex_group_fraction_digits`](@ref)

are presentation-oriented utilities.

They are useful when very long mantissas are shown and one wants clearer visual
separation of digit blocks, especially in [$\LaTeX$](https://www.latex-project.org/) output.

The grouping logic is intentionally asymmetric:

- integer parts are grouped from right to left,
- fractional parts are grouped from left to right.

---

## Design scope

This module is about **presentation**, not statistical analysis.

It does **not**:

- estimate uncertainties,
- validate the statistical meaning of an error bar,
- enforce one universal scientific-formatting convention.

Instead, it provides a practical formatting layer that is robust enough for
analysis logs and flexible enough for polished output.

---

## Practical note

Because the compact formatter uses branch logic based on the relative size of
the error and the central value, the exact output style may vary across
magnitude regimes.

That variability is intentional: the formatter prefers readability and compact
notation over rigid uniform appearance.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.AvgErrFormatter,
]
Private = true
```