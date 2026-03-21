# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADTaylorSeries

This module provides the `TaylorSeries.jl` backend for direct scalar
`n`-th-derivative evaluation inside the residual-based error-estimation stack.

---

## Overview

The direct Taylor backend expands the callable around a scalar point using a
truncated Taylor polynomial and extracts the requested derivative from that
single expansion.

Compared with repeated first-derivative application, this can be effective when
the callable is compatible with `TaylorSeries.Taylor1` arithmetic and a
moderately high derivative order is needed.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect`](@ref).

---

## Notes

- This backend is still scalar-point oriented: it returns one derivative value,
  not a full jet.
- It is an internal backend module used by derivative-based error estimators.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADTaylorSeries,
]
Private = true
```
