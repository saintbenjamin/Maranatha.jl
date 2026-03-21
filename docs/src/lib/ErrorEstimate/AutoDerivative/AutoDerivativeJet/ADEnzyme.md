# Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADEnzyme

This module provides the `Enzyme.jl` backend for derivative-jet construction
inside the residual-based error-estimation stack.

---

## Overview

The Enzyme jet backend builds

```julia
[f(x), f'(x), ..., f^(nmax)(x)]
```

by recursively applying reverse-mode differentiation to the previously
constructed scalar callable.

This gives the jet-oriented error-estimation path an Enzyme-based option while
preserving the same output layout as the other jet backends.

Backend selection is handled by
[`Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet`](@ref).

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.ErrorEstimate.AutoDerivative.AutoDerivativeJet.ADEnzyme,
]
Private = true
```
