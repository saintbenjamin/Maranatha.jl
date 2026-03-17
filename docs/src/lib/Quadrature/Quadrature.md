# Maranatha.Quadrature

`Maranatha.Quadrature` is the rule-dispatched numerical integration backend of
`Maranatha.jl`.

This page expands the higher-level design notes that are intentionally kept
shorter in the module docstring.

---

## Supported backends

### 1. [`Maranatha.Quadrature.NewtonCotes`](@ref)

Exact-rational composite Newton-Cotes construction.

Features:

- `Rational{BigInt}` exact local moment matching
- Exact global ``\beta`` coefficient assembly
- Composite boundary tiling validation
- `Float64` conversion only at final stage
- Process-local caching of assembled weights

Supported rule symbols:

```julia
:newton_p3, :newton_p4, :newton_p5, ...
```

Supported boundary patterns:

```julia
:LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX
```

### 2. [`Maranatha.Quadrature.Gauss`](@ref)

Single-interval Gauss-family rules on ``[-1,1]``:

- Gauss-Legendre
- Gauss-Radau (left or right)
- Gauss-Lobatto

Composite repetition over uniform subintervals is also supported.

Supported rule symbols:

```julia
:gauss_p2, :gauss_p3, :gauss_p4, ...
```

Boundary selects the rule family:

- `:LU_EXEX` → Legendre
- `:LU_INEX` → Radau (left)
- `:LU_EXIN` → Radau (right)
- `:LU_ININ` → Lobatto

Nodes and weights are cached per `(n, boundary)` pair.

### 3. [`Maranatha.Quadrature.BSpline`](@ref)

B-spline-based quadrature using:

- Uniform knot construction
- Greville abscissae nodes
- Exact basis integrals
- Interpolation mode
- Optional smoothing mode with a Tikhonov second-difference penalty

Supported rule symbols:

```julia
:bspline_interp_p2, :bspline_interp_p3, ...
:bspline_smooth_p2, :bspline_smooth_p3, ...
```

Boundary selection controls endpoint clamping behavior.

### 4. [`Maranatha.Quadrature.QuadratureDispatch`](@ref)

Provides the unified interface for node/weight generation and tensor-product
integration:

- [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_1d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_2d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_3d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_4d`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature_nd`](@ref)
- [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref)

---

## Computational strategy

All multidimensional integration is carried out through explicit
tensor-product construction:

```math
\sum_{i_1 , i_2 , \ldots , i_d} w_{i_1} w_{i_2} \ldots w_{i_d} f\left( x_{i_1} , x_{i_2} , \ldots , x_{i_d} \right)
```

Implementation details:

- explicit nested loops are used for $d \le 4$
- an odometer-style multi-index traversal is used for general `dim`
- zero-weight entries may be skipped for efficiency
- accumulation order is kept deterministic

The computational cost scales as:

```math
\mathcal{O} \left( \texttt{length(xs)}^{\texttt{dim}} \right)
```

---

## Boundary semantics

The shared boundary patterns

```julia
:LU_ININ  :LU_EXIN  :LU_INEX  :LU_EXEX
```

are interpreted by each backend according to its mathematical construction.
In practice, they control endpoint inclusion, family selection, or knot
clamping behavior depending on the rule family.

---

## Scope notes

`Maranatha.Quadrature` is intended as a deterministic and research-oriented
integration layer.

In particular:

- it does **not** provide adaptive quadrature
- it does **not** apply parallelism internally
- it does enforce backend-specific rule constraints explicitly
- it is aimed at structured numerical experiments rather than opaque black-box use

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature,
]
Private = true
```