# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADForwardDiff

This module provides the `ForwardDiff.jl` backend for derivative-jet
construction inside the residual-based error-estimation stack.

---

## Overview

The ForwardDiff jet backend builds the derivative sequence

```julia
[f(x), f'(x), ..., f^(nmax)(x)]
```

by repeatedly applying `ForwardDiff.derivative` to the previously constructed
scalar callable.

This backend is simple and broadly compatible, making it a practical default
for many jet-based workflows.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet`](@ref).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADForwardDiff,
]
Private = true
```
