# ============================================================================
# src/error/ErrorEstimator/estimate_error_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_1d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    ) -> Float64

Estimate the leading truncation error for a ``1``-dimensional composite Newton-Cotes rule
using the exact midpoint residual expansion derived from rational weight assembly.

# Function description
This routine computes a *model-based leading truncation term*
consistent with the exact composite Newton–Cotes construction
implemented in the `Integrate` module.

The procedure is:

1) Let ``\\displaystyle{h = \\frac{b-a}{N}}``.

2) Using the exact rational composite weights ``\\beta``, determine the
   first nonzero midpoint residual order ``k`` and its exact coefficient:
```math
\\texttt{diff}_k = \\int\\limits_0^{N_{\\text{sub}}} du \\; \\left( u - c \\right)^k - \\sum_0^{N_{\\text{sub}}} \\beta_j \\, \\left( j - c \\right)^k
```
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}
```
where:
- ``\\displaystyle{c = \\frac{N}{2}}`` is the midpoint in dimensionless coordinate,
- ``\\beta_j`` are the exact composite coefficients.

3) Evaluate the ``k``-th derivative of ``f`` at the physical midpoint:
```math
\\bar{x} = \\frac{a+b}{2}
```

4) Return the modeled leading error term:
```math
E = \\texttt{coeff} \\, h^{k+1} \\, f^{k}\\left(\\bar{x}\\right)
```

This matches the leading term of the Taylor expansion of the composite
Newton-Cotes rule around the midpoint and is fully consistent with
the exact rational assembly used in [`Maranatha.Integrate`](@ref).

# Mathematical structure
If the composite rule integrates monomials up to order ``m`` exactly,
then the first nonzero residual term appears at derivative order ``k > m``,
and the truncation error behaves like:
```math
E = C_k \\, h^{k+1} \\, f^{k}\\left(\\bar{x}\\right) + \\left( \\text{higher-order terms} \\right) \\,.
```
This routine returns exactly that leading term.

# Arguments
- `f`:
    Scalar callable integrand `f(x)` (function, closure, or callable struct).
- `a`, `b`:
    Lower and upper bounds of the integration interval.
- `N`:
    Number of subintervals.
    Must satisfy the composite tiling constraint for `(rule, boundary)`.
- `rule`:
    Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
- `boundary`:
    Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Returns
- `Float64`:
    Leading truncation error estimate.

# Errors
- Propagates any errors from:
  - composite weight assembly,
  - midpoint residual extraction,
  - derivative evaluation ([`nth_derivative`](@ref)).
- Returns `0.0` only if the detected residual order is `k == 0`
  (degenerate or pathological case).

# Notes
- This is a *leading-term asymptotic model*, not a strict upper bound.
- The coefficient is derived from exact rational arithmetic and
  converted to `Float64` only at the final stage.
"""
function estimate_error_1d(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=64)
    k == 0 && return 0.0  # practically shouldn't happen, but safe

    x̄ = (aa + bb) / 2

    d = nth_derivative(
        f,
        x̄, k;
        h=h, rule=rule, N=N, dim=1,
        side=:mid, axis=:x, stage=:midpoint
    )

    return Float64(coeffR) * h^(k+1) * d

end