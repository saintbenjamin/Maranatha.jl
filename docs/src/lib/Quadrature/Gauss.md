# Maranatha.Quadrature.Gauss

`Maranatha.Quadrature.Gauss` provides the Gauss-family backend used by
`Maranatha.Quadrature`.

## Overview

This module implements three single-interval quadrature families on $[-1,1]$:

- Gauss-Legendre
- Gauss-Radau
- Gauss-Lobatto

and also provides composite repetition over uniform subintervals.

The design goal is to keep the Gauss backend numerically transparent and explicit:

- single-interval rules are constructed first,
- composite rules are built by blockwise repetition,
- global boundary choices are translated into per-block behavior,
- repeated node/weight requests are cached.

---

## Single-interval families

### 1. Gauss-Legendre

Implemented by [`Maranatha.Quadrature.Gauss.gauss_legendre_nodes_weights_float`](@ref).

This path uses the Golub-Welsch construction:

```math
J = \mathrm{SymTridiagonal}(a,b)
```

with the Legendre Jacobi coefficients for the weight $w(x)=1$ on $[-1,1]$.
The eigenvalues of $J$ become the nodes, and the squared first components of the
normalized eigenvectors determine the weights.

This is the cleanest and usually most stable path in the module.

### 2. Gauss-Radau

Implemented by:

- [`Maranatha.Quadrature.Gauss.gauss_radau_left_nodes_weights_float`](@ref)
- [`Maranatha.Quadrature.Gauss.gauss_radau_right_nodes_weights_float`](@ref)

For Legendre weight $1$:

- left Radau includes $x=-1$, and the remaining roots satisfy
  ```math
  P_n(x) + P_{n-1}(x) = 0
  ```
- right Radau includes $x=+1$, and the remaining roots satisfy
  ```math
  P_n(x) - P_{n-1}(x) = 0
  ```

These roots are obtained by Newton iteration, seeded from Gauss-Legendre nodes.

### 3. Gauss-Lobatto

Implemented by [`Maranatha.Quadrature.Gauss.gauss_lobatto_nodes_weights_float`](@ref).

This family includes both endpoints. The interior nodes are obtained from the
zeros of $P^{\prime}_{n-1}(x)$, solved in the equivalent form

```math
P_{n-2}(x) - x P_{n-1}(x) = 0.
```

As with the Radau routines, Newton iteration is used with Legendre-based seeds.

---

## Legendre helpers

The following helpers support the root solvers:

- [`Maranatha.Quadrature.Gauss._legendre_Pn_Pn1`](@ref)
- [`Maranatha.Quadrature.Gauss._legendre_Pn_deriv`](@ref)
- [`Maranatha.Quadrature.Gauss._clamp_open`](@ref)

The derivative identity used in Newton steps contains a denominator proportional to
$x^2 - 1$, so the code intentionally clamps iterates away from $\pm 1$ for numerical
hygiene.

This is not a mathematical change to the rule itself; it is only a floating-point
stabilization device.

---

## Boundary interpretation

The Gauss backend uses the same four boundary selectors as the broader quadrature
system:

    :LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX

For the single-interval Gauss-family dispatcher:

- `:LU_EXEX` → Gauss-Legendre
- `:LU_INEX` → left Radau
- `:LU_EXIN` → right Radau
- `:LU_ININ` → Lobatto

However, in the **composite** setting, the meaning is deliberately global rather
than per-block.

That is why [`Maranatha.Quadrature.Gauss._local_boundary_for_block`](@ref) exists.

### Composite policy

Interior blocks always use Gauss-Legendre.
Only the true first and/or last block receives Radau-type endpoint treatment.

So:

- `:LU_EXEX` → all blocks Legendre
- `:LU_INEX` → first block left Radau, others Legendre
- `:LU_EXIN` → last block right Radau, others Legendre
- `:LU_ININ` → first block left Radau, last block right Radau, interior Legendre

This avoids incorrectly treating internal block boundaries as physical integration
endpoints.

---

## Composite construction

Two helpers build repeated-block Gauss grids:

- [`Maranatha.Quadrature.Gauss._composite_gauss_nodes_weights`](@ref)
- [`Maranatha.Quadrature.Gauss._composite_gauss_u_grid`](@ref)

### Physical interval version

[`Maranatha.Quadrature.Gauss._composite_gauss_nodes_weights`](@ref) maps each block from $[-1,1]$ to its physical
subinterval $[x_L, x_R]$ using the standard affine transform.

### Dimensionless version

[`Maranatha.Quadrature.Gauss._composite_gauss_u_grid`](@ref) instead builds a grid on

```math
u \in [0,N]
```

with unit blocks $[m,m+1]$. This is useful when the physical scaling from $[a,b]$
is handled elsewhere in the pipeline.

---

## Caching

The module maintains a process-local cache:

- [`Maranatha.Quadrature.Gauss._GAUSS_CACHE`](@ref)

Key:

```julia
(n, boundary)
```

Value:

```julia
(nodes::Vector{Float64}, weights::Vector{Float64})
```

This cache avoids repeated eigen-decompositions and repeated Newton solves when the
same Gauss-family configuration is requested multiple times.

The cache is not persistent across Julia sessions.

---

## Rule-symbol parsing

The Gauss backend uses the symbol convention

```julia
:gauss_p2, :gauss_p3, :gauss_p4, ...
```

where the integer encodes the number of Gauss points per block.

Helpers:

- [`Maranatha.Quadrature.Gauss._is_gauss_rule`](@ref)
- [`Maranatha.Quadrature.Gauss._parse_gauss_p`](@ref)

are provided so the higher-level dispatcher can recognize and decode these rules
cleanly.

---

## Scope notes

This module does **not** implement:

- adaptive Gauss-Kronrod style refinement,
- arbitrary orthogonal polynomial families,
- symbolic exact arithmetic analogous to the Newton-Cotes backend.

Everything here is `Float64`-based and meant to remain explicit, deterministic, and
compatible with the tensor-product quadrature layer above it.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.Gauss,
]
Private = true
```