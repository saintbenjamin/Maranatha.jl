# Maranatha.Quadrature.NewtonCotes

[`Maranatha.Quadrature.NewtonCotes`](@ref) implements the exact-rational composite
Newton-Cotes backend used by `Maranatha.jl`.

## Overview

This backend constructs composite Newton-Cotes rules by assembling local blocks
in exact rational arithmetic and converting to `Float64` only at the final stage.
It is intended for deterministic and analytically transparent quadrature studies.

Main characteristics:

- exact local moment matching with `Rational{BigInt}`
- exact global coefficient assembly for composite rules
- explicit boundary-pattern handling
- deterministic, non-adaptive construction
- cache reuse for repeated `(p, boundary, Nsub)` configurations

---

## Exact arithmetic strategy

The core exact number type is:

```julia
RBig = Rational{BigInt}
```

This is used so that:

- local monomial moments are exact,
- Vandermonde-like systems for local weights remain exact,
- overlapping composite block contributions cancel exactly when they should,
- the global coefficient vector $\beta$ is mathematically exact before conversion.

This is especially useful when studying rule structure or debugging composite
weight assembly.

---

## Supported rule form

Rule symbols are encoded as:

```julia
:newton_p3, :newton_p4, :newton_p5, ...
```

Here `p` denotes the local node count used for each Newton-Cotes block.

---

## Boundary patterns

The backend uses the shared boundary symbols:

```julia
:LU_ININ, :LU_EXIN, :LU_INEX, :LU_EXEX
```

These determine how the left and right boundary blocks are constructed.
At this layer, the boundary pattern is decoded into local block types, which
then determine the corresponding local widths and node placement rules.

---

## Local block model

A composite rule is built from local blocks on a dimensionless grid.
For a block with $p$ nodes:

- closed block width: $p - 1$
- opened block width: $p$

Local nodes are placed according to block type:

- closed: `0:(p-1)`
- opened backward: `1:p`
- opened forward: `0:(p-1)`

The local weights $\alpha$ are obtained by exact moment matching on $[0, w]$.

---

## Composite tiling constraint

The global interval is interpreted in dimensionless subinterval units and must
be tiled by:

- one left boundary block,
- `m` interior closed blocks,
- one right boundary block.

The required structure is:

```math
N_{\mathrm{sub}} = w_L + m \, (p-1) + w_R
```

where $w_L$ and $w_R$ depend on the boundary pattern.

If this condition is not satisfied, the implementation throws an error and
reports nearby valid $N_\mathrm{sub}$ values.

---

## Assembly flow

The exact assembly routine follows this structure:

1. validate `(p, boundary, Nsub)` via the tiling constraint,
2. construct the left boundary block,
3. append interior closed blocks,
4. construct the right boundary block,
5. accumulate everything into a single exact global coefficient vector $\beta$.

The final quadrature weights are then obtained by scaling with the physical
mesh size $h$.

---

## Cache behavior

To avoid repeating expensive exact-rational assembly, the module maintains a
process-local cache:

```julia
_NS_BETA_CACHE :: Dict{Tuple{Int,Symbol,Int}, Vector{Float64}}
```

The cache key is `(p, boundary, Nsub)`, and the stored value is the already
converted `Float64` coefficient vector.

This is particularly useful when repeatedly calling the same configuration
inside convergence scans or plotting workflows.

---

## Practical notes

- This backend prioritizes exactness and transparency over raw speed.
- Large `p` may produce very large exact rational coefficients and can become
  computationally heavy.
- The cache is local to the Julia process and is not serialized.
- No adaptive quadrature or parallelism is introduced at this layer.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.NewtonCotes,
]
Private = true
```