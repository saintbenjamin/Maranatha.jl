# Maranatha.Utils.JobLoggerTools

## Overview

`Maranatha.Utils.JobLoggerTools` provides lightweight logging helpers for
timestamped output, stage delimiters, assertion-style checks, and timing
messages.

The module is designed to support long-running numerical workflows where it is
useful to keep runtime logs readable without introducing heavy logging
dependencies.

---

## Logging style

All message helpers ultimately build on [`Maranatha.Utils.JobLoggerTools.println_benji`](@ref), which prints a
timestamped message and optionally prepends a job ID.

This produces a consistent logging style across the package.

Common message categories include:

- plain log lines,
- stage delimiters,
- warnings,
- informational messages,
- debug messages,
- assertion failures,
- fatal errors.

---

## Timing macro

The macro [`Maranatha.Utils.JobLoggerTools.@logtime_benji`](@ref) is the timing-oriented helper of the module.

It combines:

- `@timed` execution,
- GC allocation-difference inspection,
- timestamped logging,
- optional job ID prefixing.

This makes it convenient to profile a specific expression inline while keeping
the code syntax compact.

---

## Stage helpers

The helpers [`Maranatha.Utils.JobLoggerTools.log_stage_benji`](@ref) and 
[`Maranatha.Utils.JobLoggerTools.log_stage_sub1_benji`](@ref) are
meant for visual separation in console logs.

They do not change program state; they only structure the output so that major
and minor processing stages are easier to identify.

---

## Error and assertion helpers

[`Maranatha.Utils.JobLoggerTools.error_benji`](@ref) 
combines logging and throwing, so that the same message is
both printed and raised as an exception.

[`Maranatha.Utils.JobLoggerTools.assert_benji`](@ref) 
builds on top of that pattern by turning a failed boolean
condition into a logged assertion failure.

This is intentionally lightweight and pragmatic rather than a full-featured
logging/error framework.

---

## Design scope

This module is intentionally simple.

It does **not**:

- provide logging levels with filtering rules,
- manage external log sinks,
- implement file-based logging rotation,
- replace Julia's exception system.

Its role is to offer a compact, consistent logging vocabulary for the rest of
`Maranatha.jl`.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.JobLoggerTools,
]
Private = true
```