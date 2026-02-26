# ============================================================================
# src/error/ErrorEstimator/estimate_error_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ============================================================
# 1D error estimator legacy (leading terms)
# ============================================================

"""
    estimate_error_1d_legacy(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate the integration error for a 1D integral of `f(x)` over `[a, b]`
using rule-dependent, derivative-based *heuristics* designed to match the
lightweight, unified tensor-error philosophy used in higher dimensions.

# Function description
This estimator intentionally prioritizes:
- **speed** (few derivative evaluations),
- **stable scaling** with step size `h = (b-a)/N`,
- **consistent behavior across dimensions** (1D/2D/3D/4D),

rather than reproducing the full composite-rule truncation expansions.

In particular:
- The **closed** rules use a single midpoint derivative as a leading-term model.
- The **open-chain** rules use lightweight leading-term models:
  - some are **boundary-difference dominated** (endpoint stencil effect),
  - some use a **single midpoint derivative** with the tensor-error coefficient.

These estimates are used as error scales for fitting/extrapolation, not as
rigorous bounds.

# Arguments
- `f`: Scalar-to-scalar integrand callable `f(x)` (function, closure, or callable struct).
- `a`, `b`: Integration limits (scalars).
- `N`: Number of subintervals (must satisfy rule-specific constraints).
- `rule`: Integration rule symbol:
  - `:simpson13_close` → closed composite Simpson 1/3 (midpoint 4th-derivative heuristic)
  - `:simpson38_close` → closed composite Simpson 3/8 (midpoint 4th-derivative heuristic)
  - `:bode_close`      → closed composite Bode/Boole (midpoint 6th-derivative heuristic)
  - `:simpson13_open`  → open-chain Simpson 1/3 (boundary 3rd-derivative difference heuristic)
  - `:simpson38_open`  → open 3-point chained rule, panels of width `4h`
                          (midpoint 4th-derivative heuristic with coefficient `14/45`)
  - `:bode_open`       → open-chain Boole-type rule (boundary 5th-derivative difference heuristic)

# Returns
- A `Float64` heuristic error estimate (signed), interpreted as an estimate of
  `(exact - quadrature)` in the same sign convention as the implemented formula.
  If `rule` is not recognized, returns `0.0`.

# Errors
- Throws an error if `N` violates rule-specific constraints.
"""
function estimate_error_1d_legacy(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)
    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xj(j::Int) = aa + j * h

    if rule == :simpson13_close
        # closed composite Simpson 1/3 (leading term heuristic)
        N % 2 == 0 || JobLoggerTools.error_benji("Close composite Simpson 1/3 rule requires N divisible by 2, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 180.0) * h^4 * d4

    elseif rule == :simpson38_close
        # closed composite Simpson 3/8 (leading term heuristic)
        N % 3 == 0 || JobLoggerTools.error_benji("Close composite Simpson 3/8 rule requires N divisible by 3, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 80.0) * h^4 * d4

    elseif rule == :bode_close
        # closed composite Bode (leading term heuristic)
        N % 4 == 0 || JobLoggerTools.error_benji("Close composite Boole's rule requires N divisible by 4, got N = $N")
        x̄ = (aa + bb) / 2
        d6 = nth_derivative(f, x̄, 6)
        return -((2.0 / 945.0) * (bb - aa)) * h^6 * d6

    elseif rule == :simpson13_open
        (N % 2 == 0) || JobLoggerTools.error_benji("Open composite Simpson 1/3 rule requires N even, got N = $N")
        (N >= 8)     || JobLoggerTools.error_benji("Open composite Simpson 1/3 rule requires N ≥ 8, got N = $N")

        # Leading-term error model consistent with the open-chain expansion:
        #   E ≈ -(3/8) h^4 [ f'''( (x1+x2)/2 ) - f'''( (x_{N-2}+x_{N-1})/2 ) ]
        #
        # Here E is an estimate of (exact - quadrature).

        xL  = aa + 1.5 * h           # (x1 + x2)/2
        xR  = aa + (N - 1.5) * h     # (x_{N-2} + x_{N-1})/2

        d3L = nth_derivative(f, xL, 3)
        d3R = nth_derivative(f, xR, 3)

        return -(3.0 / 8.0) * h^4 * (d3L - d3R)

    elseif rule == :simpson38_open
        # IMPORTANT:
        # This is NOT the classical Simpson 3/8 composite rule.
        # This is the endpoint-free open 3-point Newton–Cotes chained rule (panel width = 4h).
        #
        # Lightweight heuristic (consistent with the "rollback" philosophy):
        # use a single midpoint 4th-derivative leading term (order h^4 scaling),
        # matching the tensor-error coefficient mapping (C = 14/45, m = 4).
        N % 4 == 0 || JobLoggerTools.error_benji("Open composite Simpson 3/8 rule requires N divisible by 4, got N = $N")
        N >= 4     || JobLoggerTools.error_benji("Open composite Simpson 3/8 rule requires N ≥ 4, got N = $N")

        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)

        return (14.0 / 45.0) * (bb - aa) * h^4 * d4

    elseif rule == :bode_open
        (N % 4 == 0) || JobLoggerTools.error_benji("Open composite Boole's rule requires N divisible by 4, got N = $N")
        (N >= 16)    || JobLoggerTools.error_benji("Open composite Boole's rule requires N ≥ 16, got N = $N")

        # Boundary-dominant leading-term model (Simpson 1/3 open-chain style):
        #   E_lead ≈ -(95/288) h^6 [ f^(5)((x2+x3)/2) - f^(5)((x_{N-3}+x_{N-2})/2) ]

        xL = (xj(2) + xj(3)) / 2
        xR = (xj(N-3) + xj(N-2)) / 2

        d5L = nth_derivative(f, xL, 5)
        d5R = nth_derivative(f, xR, 5)

        return -(95.0 / 288.0) * h^6 * (d5L - d5R)

    else
        return 0.0
    end
end

# ============================================================
# 1D error estimator (leading terms)
# ============================================================

"""
    estimate_error_1d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol
    ) -> Float64

Return a fast ``1``-dimensional quadrature error *scale* for 
```math
\\int\\limits_{a}^{b} \\; dx \\; f(x)
```
using a lightweight derivative-based heuristic. 

For selected opened composite rules, a boundary-difference
proxy is used. 
Derivatives are attempted with [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) first, 
with an automatic
[`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback when non-finite (`Inf`/`NaN`) values occur.

# Function description
This routine provides a **cheap, consistent error scale proxy** intended to:
- supply per-point weights ``\\sigma(h)`` for least-``\\chi^2``-fitting ([`Maranatha.FitConvergence.fit_convergence`](@ref)), and
- match the same ``h``-scaling convention used by the multidimensional error
  estimators in this package.

It is *not* a rigorous truncation bound and does not attempt to reproduce the
full composite-rule error expansion.

Two regimes are supported:

## (A) Boundary-difference model (for selected opened composite rules)
For rules flagged by [`_has_boundary_error_model`](@ref)`(rule)`, the estimator uses a leading
boundary-difference proxy of the form
```math
E \\approx \\texttt{K} \\, h^\\texttt{p} \\, ( D_L - D_R ) \\,
```

where
- ``\\displaystyle{h = \\frac{b-a}{N}}``,
- ``x_L = a + \\texttt{z} \\, h``,
- ``x_R = a + ( N - \\texttt{z} ) \\, h``,
- `(p, K, m, z) =`[`_boundary_error_params`](@ref)`(rule)`.

This branch is designed for (endpoint-free) opened composite rule formulas whose leading error
behavior can be boundary-dominant, and often improves least ``\\chi^2`` fitting stability for those rules.

### Derivative evaluation and [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback
All derivatives in this routine are evaluated via the internal helper `_nth_deriv_safe`:
1) compute using [`nth_derivative`](@ref) ([`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)-based),
2) if non-finite, emit a [`Maranatha.JobLoggerTools.warn_benji`](@ref) and retry with [`nth_derivative_taylor`](@ref) ([`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl)-based),
3) throw an error only if the [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback is also non-finite.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, the estimator follows the same *single-sample midpoint derivative* pattern
used in the multidimensional error estimators:

1) Obtain `(m, C)` via [`_rule_params_for_tensor_error`](@ref)`(rule)`.
2) Build `1`-dimensional nodes/weights `(xs, wx)` via [`Maranatha.Integrate.quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule)`.
3) Evaluate ``f^{(m)}(\\bar{x})`` once at the midpoint ``\\displaystyle{\\bar{x} = \\frac{a+b}{2}}`` (with the same fallback logic).
4) Form the weight-sum prox ``\\displaystyle{I = \\left( \\sum_j \\texttt{wx[j]} \\right) \\ast f^{(m)}\\left( \\bar{x} \\right)}``.
   (Since the derivative sample is constant across nodes, this is equivalent to
   accumulating ``\\displaystyle{\\left( \\sum_j \\texttt{wx[j]} \\right) \\ast f^{(m)}\\left( \\bar{x} \\right)}``.)
5) Return ``E = C \\; (b - a) \\; h^m \\; I``.

# Arguments
- `f`: Scalar callable integrand `f(x)` (function, closure, or callable struct).
- `a`, `b`: Integration limits.
- `N`: Number of subintervals defining ``\\displaystyle{h = \\frac{b-a}{N}}``.
- `rule`: Quadrature rule symbol.

# Returns
- `Float64`: A heuristic (signed) error scale proxy. If `m == 0` for the selected
  `rule`, returns `0.0`.

# Notes
- Some rules may have **negative quadrature weights**. This estimator
  intentionally preserves the rule-defined weight sum ``\\displaystyle{\\sum_j \\texttt{wx[j]}}``, rather than enforcing
  any normalization.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced explicitly in the boundary-model branch for supported open rules, and
  - enforced in [`Maranatha.Integrate.quadrature_1d_nodes_weights`](@ref) for the default midpoint path.
- The [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback requires the integrand to accept generic number types
  (e.g. `Taylor1`). If the integrand dispatch is restricted to `Real` only, the fallback
  may raise a `MethodError`.

# Errors
- Throws an error if `(N, rule)` violates rule constraints.
- Throws an error if both [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) derivatives are non-finite in the
  selected estimator branch.
"""
function estimate_error_1d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    @inline function _nth_deriv_safe(g, x, n; side::Symbol=:mid)
        d = nth_derivative(g, x, n)
        if !isfinite(d)
            JobLoggerTools.warn_benji(
                "Non-finite derivative (ForwardDiff); trying Taylor fallback " *
                "h=$h x=$x n=$n rule=$rule N=$N side=$side"
            )
            d = nth_derivative_taylor(g, x, n)
            if !isfinite(d)
                JobLoggerTools.error_benji(
                    "Non-finite in 1D error estimator even after Taylor fallback: " *
                    "h=$h x=$x deriv=$d n=$n rule=$rule N=$N side=$side"
                )
            end
        end
        return d
    end

    # ---- special boundary-difference models ----
    if _has_boundary_error_model(rule)
        if rule == :simpson13_open
            (N % 2 == 0) || JobLoggerTools.error_benji("open composite Simpson 1/3 rule requires N even, got N = $N")
            (N >= 8)     || JobLoggerTools.error_benji("open composite Simpson 1/3 rule requires N ≥ 8, got N = $N")
        elseif rule == :bode_open
            (N % 4 == 0) || JobLoggerTools.error_benji("Open composite Boole's rule requires N divisible by 4, got N = $N")
            (N >= 16)    || JobLoggerTools.error_benji("Open composite Boole's rule requires N ≥ 16 (non-overlapping end stencils), got N = $N")
        end

        p, K, dord, off = _boundary_error_params(rule)
        xL = aa + off*h
        xR = aa + (N-off)*h

        # dL = nth_derivative(f, xL, dord)
        # dR = nth_derivative(f, xR, dord)
        dL = _nth_deriv_safe(f, xL, dord; side=:L)
        dR = _nth_deriv_safe(f, xR, dord; side=:R)

        return K * h^p * (dL - dR)
    end

    # ---- default tensor-style midpoint model ----
    x̄ = (aa+bb)/2

    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)

    # d = nth_derivative(f, x̄, m)
    d = _nth_deriv_safe(f, x̄, m; side=:mid)

    # I = 0.0
    # @inbounds for j in eachindex(xs)
    #     I += wx[j] * d
    # end
    I = d * sum(wx)

    return C*(bb-aa)*h^m*I
end