# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADFastDifferentiation

This module provides the `FastDifferentiation.jl` backend for derivative-jet
construction inside the residual-based error-estimation stack.

---

## Overview

The backend traces the callable symbolically, constructs the derivative chain
from order `0` through `nmax`, compiles the resulting expressions, and returns
the full jet at the target point.

It is most useful when the callable is compatible with symbolic tracing and a
single compiled symbolic graph is preferable to repeated AD evaluation.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet`](@ref).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADFastDifferentiation,
]
Private = true
```
