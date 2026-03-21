# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADFastDifferentiation

This module provides the `FastDifferentiation.jl` backend for direct scalar
`n`-th-derivative evaluation inside the residual-based error-estimation stack.

---

## Overview

The backend traces the scalar callable on a symbolic variable, differentiates
the symbolic expression to the requested order, compiles the result, and
evaluates that compiled derivative at the target point.

This can be attractive for algebraic callables that trace cleanly into
`FastDifferentiation`, especially when symbolic reuse is preferable to repeated
AD calls.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect`](@ref).

---

## Notes

- The callable must accept FastDifferentiation symbolic inputs.
- This is an internal backend module used by derivative-based error
  estimators.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADFastDifferentiation,
]
Private = true
```
