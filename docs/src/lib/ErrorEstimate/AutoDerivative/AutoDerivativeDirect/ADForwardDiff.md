# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADForwardDiff

This module provides the `ForwardDiff.jl` backend for direct scalar
`n`-th-derivative evaluation inside the residual-based error-estimation stack.

---

## Overview

The direct ForwardDiff backend constructs higher-order derivatives by repeated
application of `ForwardDiff.derivative` to nested scalar callables.

It is the most generally useful direct backend in the current package because
it works well for many ordinary Julia scalar callables without requiring
symbolic tracing or Taylor-series compatibility.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect`](@ref).

---

## Notes

- The callable need not be declared as `::Function`; closures and callable
  structs are also accepted.
- This is an internal backend module used by derivative-based error
  estimators.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADForwardDiff,
]
Private = true
```
