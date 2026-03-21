# Maranatha.Quadrature.BSpline

`Maranatha.Quadrature.BSpline` provides a spline-based deterministic quadrature
backend built from uniform knot vectors, Greville abscissae, exact basis
integrals, and either interpolation- or smoothing-based weight construction.

It plays a different role from the exact Newton-Cotes and Gauss-family backends:
rather than assembling classical quadrature formulas, it builds a spline space,
samples the integrand at spline-associated nodes, and turns spline basis
information into quadrature weights.

---

## Rule symbols

Supported rule prefixes are:

- `:bspline_interp_p2`, `:bspline_interp_p3`, ...
- `:bspline_smooth_p2`, `:bspline_smooth_p3`, ...

The helper logic splits a rule into:

- kind: `:interp` or `:smooth`
- degree: `p`

---

## High-level construction

For a given interval $[a,b]$, subdivision count $N$, degree $p$, and boundary
mode:

1. build a uniform knot vector,
2. apply endpoint clamping according to the selected boundary mode,
3. compute Greville abscissae,
4. evaluate the full B-spline basis at those nodes,
5. compute exact basis integrals,
6. convert that information into quadrature weights.

The final rule has the form

```math
\int_a^b dx \; f(x) \approx \sum_j w_j \, f(x_j).
```

---

## Knot construction policy

The knot vector is built from a uniform step size

```math
h = \frac{b-a}{N}
```

using a simple extended grid, followed by endpoint clamping.

The low-level knot helper machinery can interpret boundary symbols as:

- `:LU_ININ` : clamp both endpoints
- `:LU_INEX` : clamp left endpoint only
- `:LU_EXIN` : clamp right endpoint only
- `:LU_EXEX` : no endpoint clamping

In the current public quadrature driver, however, B-spline node/weight
construction is intentionally restricted to:

- `boundary == :LU_ININ`

So the internal knot logic is broader than the currently enabled public policy.

---

## Greville nodes

Greville abscissae are used as quadrature nodes because they provide a standard,
stable collocation choice for spline-based constructions.

- For $p \ge 1$, they are averages of consecutive interior knots.
- For $p = 0$, the implementation uses knot-span midpoints.

---

## Basis evaluation

All basis values are computed with the Cox-de Boor recursion.

Implementation policy:

- start from degree-$0$ indicator functions,
- elevate degree step by step,
- treat `x == t[end]` using the conventional last-basis endpoint rule.

This is a deterministic, explicit implementation meant for transparency rather
than for a fully general spline library API.

---

## Exact basis integrals

The basis integrals are computed analytically from the knot vector. This is one
of the central reasons the construction is useful for quadrature:

- spline basis evaluation supplies the collocation structure,
- exact basis integrals supply the target linear functional.

---

## Interpolation mode

For `kind == :interp`, the method assumes spline interpolation at the Greville
nodes.

If `A` is the collocation matrix and `b` is the vector of basis integrals, then
the quadrature weights are obtained from the transposed solve

```math
w = A^{-T} b.
```

In the implementation this appears as a solve against [`transpose(A)`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.transpose).

---

## Smoothing mode

For `kind == :smooth`, the method uses a Tikhonov-style discrete penalty based
on second differences of spline coefficients.

With penalty matrix $R$ and smoothing strength $\lambda$, the method builds a
regularized system of the form

```math
A^T A + \lambda R.
```

The resulting quadrature weights are then assembled from the corresponding
regularized linear solve.

This is a practical smoothing construction, not a claim of exact equivalence to
the continuous roughness penalty

```math
\int dx \; (s''(x))^2.
```

---

## Singular-safe solve policy

Some spline collocation systems can be singular or effectively singular,
especially under certain boundary or smoothing configurations.

To avoid hard failure, the implementation uses a robust helper:

- first try the standard dense solve,
- if that fails with [`LinearAlgebra.SingularException`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SingularException), fall back to an
  SVD-based pseudo-inverse solve,
- return the minimum-norm solution on the retained singular subspace.

This keeps the quadrature pipeline usable in borderline configurations.

---

## Scope and limitations

This backend is intentionally pragmatic.

It is designed for:

- deterministic quadrature experiments,
- transparent basis-based weight construction,
- simple interpolation / smoothing comparisons.

It is not intended as:

- a full-featured spline package,
- an adaptive quadrature engine,
- a highly optimized large-scale spline collocation framework.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.BSpline,
]
Private = true
```
