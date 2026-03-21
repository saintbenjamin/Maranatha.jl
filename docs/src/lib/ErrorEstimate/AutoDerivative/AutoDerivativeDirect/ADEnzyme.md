# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADEnzyme

This module provides the `Enzyme.jl` backend for direct scalar
`n`-th-derivative evaluation inside the residual-based error-estimation stack.

---

## Overview

The direct Enzyme backend builds higher-order scalar derivatives by repeatedly
wrapping the current callable in a first-derivative `Enzyme.gradient` call.

Its role is narrow:

- accept a scalar callable `f`,
- evaluate `f^(n)(x)` at one scalar point,
- return that value in the scalar type of `x`.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect`](@ref).

---

## Practical note

This backend is mainly useful as an experimental or comparative path. The
default practical backend in typical workflows is usually
[`ADForwardDiff`](@ref Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADForwardDiff).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeDirect.ADEnzyme,
]
Private = true
```
