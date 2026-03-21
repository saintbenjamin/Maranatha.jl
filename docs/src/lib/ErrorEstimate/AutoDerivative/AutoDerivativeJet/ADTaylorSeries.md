# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADTaylorSeries

This module provides the `TaylorSeries.jl` backend for derivative-jet
construction inside the residual-based error-estimation stack.

---

## Overview

The Taylor backend expands the callable around the evaluation point and
extracts the function value and all derivatives up to `nmax` from the same
truncated Taylor representation.

This is often effective when the callable is compatible with
`TaylorSeries.Taylor1` arithmetic and many derivative orders are required at
one point.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet`](@ref).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADTaylorSeries,
]
Private = true
```
