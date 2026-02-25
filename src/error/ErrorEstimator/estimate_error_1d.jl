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
        N % 2 == 0 || error("Simpson 1/3 requires N divisible by 2, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 180.0) * h^4 * d4

    elseif rule == :simpson38_close
        # closed composite Simpson 3/8 (leading term heuristic)
        N % 3 == 0 || error("Simpson 3/8 requires N divisible by 3, got N = $N")
        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)
        return -((bb - aa) / 80.0) * h^4 * d4

    elseif rule == :bode_close
        # closed composite Bode (leading term heuristic)
        N % 4 == 0 || error("Bode's rule requires N divisible by 4, got N = $N")
        x̄ = (aa + bb) / 2
        d6 = nth_derivative(f, x̄, 6)
        return -((2.0 / 945.0) * (bb - aa)) * h^6 * d6

    elseif rule == :simpson13_open
        (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
        (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")

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
        N % 4 == 0 || error("Open 3-point chained rule requires N divisible by 4, got N = $N")
        N >= 4     || error("Open 3-point chained rule requires N ≥ 4, got N = $N")

        x̄ = (aa + bb) / 2
        d4 = nth_derivative(f, x̄, 4)

        return (14.0 / 45.0) * (bb - aa) * h^4 * d4

    elseif rule == :bode_open
        (N % 4 == 0) || error("Bode open-chain (open composite Boole) requires N divisible by 4, got N = $N")
        (N >= 16)    || error("Bode open-chain (open composite Boole) requires N ≥ 16, got N = $N")

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

Estimate a 1D integration error *scale* for `∫_a^b f(x) dx` using a lightweight
derivative-based heuristic consistent with the tensor-product philosophy used in
`estimate_error_2d/3d/4d`, with optional boundary-difference handling for
selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for:
- stabilizing least-χ² fits in convergence extrapolation, and
- providing a consistent `h`-scaling proxy across 1D–4D.

It is **not** a rigorous truncation bound and does not attempt to reproduce the
full composite-rule error expansion.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator uses a
boundary-difference leading-term model:

`E ≈ K * h^p * ( f^(dord)(xL) - f^(dord)(xR) )`,

with
- `h = (b-a)/N`,
- `xL = a + off*h`,
- `xR = a + (N-off)*h`,
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This is designed to reflect the boundary-dominant leading behavior of certain
endpoint-free chained formulas, and empirically improves stability in χ²-based
fits for those rules.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine follows the “single-sample midpoint derivative” pattern
shared with the higher-dimensional estimators:

1. Select derivative order `m` and coefficient `C` via `_rule_params_for_tensor_error(rule)`.
2. Build 1D quadrature nodes/weights `(xs, ws)` via `quadrature_1d_nodes_weights(a, b, N, rule)`.
3. Evaluate `f^(m)` **once** at the midpoint `x̄ = (a+b)/2`.
4. Accumulate `I = Σ ws[j] * f^(m)(x̄)` (preserving loop/accumulation style).
5. Return the scale model `E ≈ C * (b-a) * h^m * I`.

# Arguments
- `f`: Scalar-to-scalar callable integrand `f(x)` (function, closure, or callable struct).
- `a`, `b`: Integration limits.
- `N`: Number of subintervals defining `h = (b-a)/N` and the rule grid.
- `rule`: Integration rule symbol.

Supported rule symbols (current mapping):
- `:simpson13_close`, `:simpson38_close`, `:bode_close`
- `:simpson13_open`,  `:simpson38_open`,  `:bode_open`

# Returns
- `Float64`: heuristic signed error estimate (scale model).
  If `rule` is not recognized by `_rule_params_for_tensor_error` (and no boundary
  model is defined), returns `0.0`.

# Notes
- Some open-chain rules may involve **negative** weights in their quadrature
  formulas. This estimator intentionally preserves the weight-loop accumulation
  style, rather than enforcing normalization.
- Any rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path, and
  - enforced explicitly in the boundary-model branch for the supported open rules.

# Errors
- Throws an error if `(N, rule)` violates the rule constraints.
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

    # ---- special boundary-difference models ----
    if _has_boundary_error_model(rule)
        if rule == :simpson13_open
            (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
            (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")
        elseif rule == :bode_open
            (N % 4 == 0) || error("Open composite Boole requires N divisible by 4, got N = $N")
            (N >= 16)    || error("Open composite Boole requires N ≥ 16 (non-overlapping end stencils), got N = $N")
        end

        p, K, dord, off = _boundary_error_params(rule)
        xL = aa + off*h
        xR = aa + (N-off)*h

        dL = nth_derivative(f, xL, dord)
        dR = nth_derivative(f, xR, dord)

        return K * h^p * (dL - dR)
    end

    # ---- default tensor-style midpoint model ----
    x̄ = (aa+bb)/2

    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)

    d = nth_derivative(f, x̄, m)

    I = 0.0
    @inbounds for j in eachindex(xs)
        I += wx[j] * d
    end

    return C*(bb-aa)*h^m*I
end