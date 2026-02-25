# ============================================================================
# src/error/ErrorEstimator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorEstimator

using TaylorSeries
using Enzyme
using ForwardDiff
using LinearAlgebra
using ..Integrate
using ..BodeRule_MinOpen_MaxOpen

export estimate_error

# ============================================================
# Internal helpers (must preserve numerical behavior)
# ============================================================

"""
    nth_derivative_taylor(
        f,
        x::Real,
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using a Taylor expansion via `TaylorSeries.jl`.

# Function description
This routine evaluates the Taylor expansion of `f(x + t)` around `x`
up to order `n` using a `Taylor1` expansion variable.  
The `n`-th derivative is obtained from the `n`-th Taylor coefficient
multiplied by `n!`.

Unlike the ForwardDiff implementation, this method performs **higher-order
differentiation in a single pass** rather than recursively applying
first derivatives. It is useful for benchmarking alternative AD strategies
and for testing high-order derivative extraction based on truncated
power-series arithmetic.

This function accepts any callable object `f`, including:
- ordinary functions,
- anonymous closures,
- callable structs (functors).

# Arguments
- `f`: Scalar-to-scalar callable (`f(x)::Number` expected).
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order (must be nonnegative).

# Returns
- The `n`-th derivative value `f^(n)(x)`.

# Notes
- Internally converts `x` to `Float64` to match the surrounding numeric policy.
- This method may allocate significantly more memory than ForwardDiff,
  especially when used inside large loops or with high expansion orders.
- Intended primarily for **benchmarking and experimental comparison**, not
  necessarily for production performance in the current workflow.
"""
@inline function nth_derivative_taylor(f, x::Real, n::Int)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    xx = Float64(x)

    n == 0 && return f(xx)

    t = Taylor1(Float64, n)     # expansion variable (order n)
    y = f(xx + t)               # y is Taylor1 (or compatible)
    return y[n] * factorial(n)  # nth derivative at x
end

"""
    nth_derivative_enzyme(
        f,
        x::Real,
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using repeated reverse-mode differentiation via `Enzyme.jl`.

# Function description
This routine constructs a nested closure chain of length `n`, where each step
applies `Enzyme.gradient` in reverse mode to obtain a first derivative.
The resulting callable is then evaluated at `x`.

This mirrors the structure of the ForwardDiff-based implementation but replaces
forward-mode differentiation with Enzyme's reverse-mode AD. It is intended
primarily for benchmarking and experimentation with Enzyme in scalar
high-order differentiation contexts.

Supported callable types include:
- ordinary functions,
- anonymous closures,
- callable structs (functors).

# Arguments
- `f`: Scalar-to-scalar callable (`f(x)::Number` expected).
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order (must be nonnegative).

# Returns
- The `n`-th derivative value `f^(n)(x)`.

# Notes
- Reverse-mode AD is typically advantageous for many-input/one-output problems.
  For repeated scalar higher-order derivatives, performance may be worse than
  ForwardDiff due to closure nesting and gradient reconstruction overhead.
- This implementation intentionally preserves the closure-based structure
  for fair benchmarking against other approaches.
- Inputs are converted to `Float64` to match surrounding numeric conventions.
- Provided as a **benchmarking reference implementation**, not as the
  recommended production path in the current codebase.
"""
function nth_derivative_enzyme(
    f,
    x::Real,
    n::Int
)
    g = f
    for _ in 1:n
        prev = g
        g = t -> only(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
        # or: g = t -> first(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
    end
    return g(float(x))
end

"""
    nth_derivative(
        f, 
        x::Real, 
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using repeated `ForwardDiff.derivative`.

# Function description
This routine is intentionally written to accept any **callable** object `f`,
not only subtypes of `Function`. This includes:
- ordinary functions,
- anonymous closures,
- callable structs (functors) such as preset integrands.

This design is required for compatibility with the integrand registry and
preset-style callable wrappers while preserving ForwardDiff-based behavior.

# Arguments
- `f`: Scalar-to-scalar callable (e.g., `f(x)::Number`).
- `x::Real`: Point at which the derivative is evaluated.
- `n::Int`: Derivative order (nonnegative integer).

# Returns
- The `n`-th derivative value `f^(n)(x)`.

# Notes
- This implementation constructs a nested closure chain of length `n` and then
  evaluates it at `x`. This intentionally matches the original behavior.
- Type restriction `f::Function` is intentionally avoided because callable
  structs are not subtypes of `Function`, but must be supported.
"""
function nth_derivative(
    f, 
    x::Real, 
    n::Int
)
    g = f
    for _ in 1:n
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
    end
    return g(x)
end

"""
    _rule_params_for_tensor_error(
        rule::Symbol
    )

Map `rule` to the derivative order `m` and coefficient `C` used by the
tensor-product derivative-based error heuristics in 2D/3D/4D estimators.

# Arguments
- `rule`: Integration rule symbol.

# Returns
- `(m, C)` where:
  - `m::Int` is the derivative order used in the estimator,
  - `C` is the rule-dependent coefficient (kept as the same literal type
    as the original implementation).

If `rule` is not supported, returns `(0, 0.0)`.
"""
function _rule_params_for_tensor_error(
    rule::Symbol
)
    # IMPORTANT: keep the exact literals/types consistent with the original code.
    if rule == :simpson13_close
        return (4, -1/180)
    elseif rule == :simpson38_close
        return (4, -1/80)
    elseif rule == :bode_close
        return (6, -2/945)
    elseif rule == :simpson13_open
        return (3, -3/8)
    elseif rule == :simpson38_open
        return (4, 14/45)
    elseif rule == :bode_open
        return (6, 1.0)
    else
        return (0, 0.0)
    end
end

# ============================================================
# Boundary-difference models for open-chain rules (1D–4D)
# ============================================================

"""
    _has_boundary_error_model(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` uses a boundary-difference leading-term error model.

# Function description
For some endpoint-free (“open-chain”) rules, the dominant truncation behavior is
often controlled by **boundary corrections** rather than a purely interior
(midpoint) derivative sample.
%
This helper identifies the rule symbols for which the error estimators
(`estimate_error_1d/2d/3d/4d`) should switch from the default midpoint-based
tensor heuristic to a boundary-difference model.

# Arguments
- `rule::Symbol`: Integration rule symbol.

# Returns
- `Bool`: `true` if a boundary-difference model is defined for `rule`,
  otherwise `false`.

# Notes
- Currently enabled rules:
  - `:simpson13_open`
  - `:bode_open`
- All other rules fall back to `_rule_params_for_tensor_error(rule)`-based
  midpoint/tensor heuristics.
"""
@inline function _has_boundary_error_model(
    rule::Symbol
)::Bool
    return (rule == :simpson13_open) || (rule == :bode_open)
end

"""
    _boundary_error_params(
        rule::Symbol
    ) -> (p, K, dord, off)

Return parameters for the boundary-difference **leading-term** error model
associated with `rule`.

# Function description
This routine provides a compact parameterization for boundary-difference error
heuristics used by open-chain rules in 1D–4D estimators.

The model is expressed in the form

`E ≈ K * h^p * ( D_left - D_right )`,

where
- `h = (b-a)/N`,
- `D_left  = f^(dord)(xL)` (or an axis-wise derivative in higher dimensions),
- `D_right = f^(dord)(xR)`,

and the evaluation points are placed symmetrically near both ends:

- `xL = a + off*h`
- `xR = a + (N-off)*h`

This parameterization allows 2D/3D/4D estimators to reuse the same boundary logic
by applying the axis-wise boundary difference while integrating over the other
coordinates via the quadrature weights.

# Arguments
- `rule::Symbol`: Integration rule symbol.

# Returns
- `(p, K, dord, off)` where:
  - `p::Int`      : leading power of `h` (i.e., the model scales as `h^p`),
  - `K::Float64`  : prefactor multiplying the boundary difference,
  - `dord::Int`   : derivative order used in the boundary difference,
  - `off::Float64`: offset (in units of `h`) used to define boundary sample points
    `xL = a + off*h` and `xR = a + (N-off)*h`.

If `rule` is not supported, returns `(0, 0.0, 0, 0.0)`.

# Notes
- This is a **heuristic leading-term model** used to set a stable error *scale*
  for fitting/extrapolation. It is not a rigorous truncation bound.
- The numerical constants are chosen to match the open-chain rule expansions
  used in this project:
  - `:simpson13_open` uses a third-derivative boundary difference with `h^4`.
  - `:bode_open`     uses a fifth-derivative boundary difference with `h^6`.
"""
function _boundary_error_params(
    rule::Symbol
)
    if rule == :simpson13_open
        # E ≈ -(3/8) h^4 [ f'''(a+1.5h) - f'''(a+(N-1.5)h) ]
        return (4, -(3.0/8.0), 3, 1.5)
    elseif rule == :bode_open
        # E ≈ -(95/288) h^6 [ f^(5)(a+2.5h) - f^(5)(a+(N-2.5)h) ]
        return (6, -(95.0/288.0), 5, 2.5)
    else
        return (0, 0.0, 0, 0.0)
    end
end

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

# ============================================================
# 2D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_2d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 2D integration error *scale* over the square domain `[a,b] × [a,b]`
using a derivative-based tensor-product heuristic, with optional boundary-
difference handling for selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for stabilizing
convergence fits / extrapolation, not a rigorous truncation bound.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies
a boundary-difference leading-term model **axis by axis**:

- Along the `x`-axis, integrate over `y`:
  `I_x = ∫ [∂_x^{dord} f(xL, y) - ∂_x^{dord} f(xR, y)] dy`
- Along the `y`-axis, integrate over `x`:
  `I_y = ∫ [∂_y^{dord} f(x, yL) - ∂_y^{dord} f(x, yR)] dx`

The final model is

`E ≈ K * h^p * (I_x + I_y)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `yL, yR`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This reflects the boundary-dominant leading behavior of certain endpoint-free
chained rules, and is used to improve stability in χ²-based convergence fits.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine uses the same tensor-product “single-sample midpoint”
structure as the legacy estimator:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate two axis-wise contributions:
  1) `I_x = ∫ ∂_x^m f(x̄, y) dy` by sampling `∂_x^m` at `x̄` and integrating over `y`.
  2) `I_y = ∫ ∂_y^m f(x, ȳ) dx` by sampling `∂_y^m` at `ȳ` and integrating over `x`.
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 2D integrand callable `f(x, y)` (function, closure, or callable struct).
- `a`, `b`: Square domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Integration rule symbol (same family as in `estimate_error_1d`).

# Returns
- `Float64`: heuristic signed error estimate (scale model). If `rule` is not
  recognized (and no boundary model is defined), returns `0.0`.

# Notes
- This estimator intentionally matches the loop structure and accumulation style
  of the original implementation for reproducibility.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path, and
  - enforced explicitly in the boundary-model branch for supported open rules.
"""
function estimate_error_2d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)

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
        yL = xL
        yR = xR

        # X-axis boundary difference, integrated over y
        I1 = 0.0
        @inbounds for j in eachindex(xs)
            y = xs[j]
            gx(x) = f(x, y)
            I1 += wx[j] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
        end

        # Y-axis boundary difference, integrated over x
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            gy(y) = f(x, y)
            I2 += wx[i] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
        end

        return K * h^p * (I1 + I2)
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    I1 = 0.0
    @inbounds for j in eachindex(xs)
        y = xs[j]
        gx(x) = f(x,y)
        I1 += wx[j]*nth_derivative(gx, x̄, m)
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        gy(y) = f(x,y)
        I2 += wx[i]*nth_derivative(gy, ȳ, m)
    end

    return C*(bb-aa)*h^m*(I1 + I2)
end

# ============================================================
# 3D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_3d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 3D integration error *scale* over the cube domain `[a,b]^3`
using a derivative-based tensor-product heuristic, with optional boundary-
difference handling for selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for stabilizing
convergence fits / extrapolation, not a rigorous truncation bound.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies
a boundary-difference leading-term model **axis by axis**.

For each axis, take a boundary difference of the corresponding axis-wise
derivative and integrate over the remaining coordinates:

- X-axis:
  `I_x = ∬ [∂_x^{dord} f(xL, y, z) - ∂_x^{dord} f(xR, y, z)] dy dz`
- Y-axis:
  `I_y = ∬ [∂_y^{dord} f(x, yL, z) - ∂_y^{dord} f(x, yR, z)] dx dz`
- Z-axis:
  `I_z = ∬ [∂_z^{dord} f(x, y, zL) - ∂_z^{dord} f(x, y, zR)] dx dy`

The final model is

`E ≈ K * h^p * (I_x + I_y + I_z)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `y`, `z`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This boundary-difference structure is designed to reflect the leading behavior
of certain endpoint-free chained formulas and improve χ² stability.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine follows the legacy midpoint tensor heuristic:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate three axis-wise contributions by integrating the `m`-th derivative
  along one axis while fixing the remaining coordinates at quadrature nodes:
  - `I_x = ∬ ∂_x^m f(x̄, y, z) dy dz`
  - `I_y = ∬ ∂_y^m f(x, ȳ, z) dx dz`
  - `I_z = ∬ ∂_z^m f(x, y, z̄) dx dy`
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y + I_z)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 3D integrand callable `f(x, y, z)` (function, closure, or callable struct).
- `a`, `b`: Cube domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Integration rule symbol.

# Returns
- `Float64`: heuristic signed error estimate (scale model). If `rule` is not
  recognized (and no boundary model is defined), returns `0.0`.

# Notes
- This estimator intentionally matches the loop structure and accumulation style
  of the original implementation for reproducibility.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path, and
  - enforced explicitly in the boundary-model branch for supported open rules.
"""
function estimate_error_3d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx

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
        yL = xL; yR = xR
        zL = xL; zR = xR

        # X-axis: ∬ [∂_x^(dord) f(xL,y,z) - ∂_x^(dord) f(xR,y,z)] dy dz
        I1 = 0.0
        @inbounds for j in eachindex(ys)
            y = ys[j]
            wyj = wy[j]
            for k in eachindex(zs)
                z = zs[k]
                gx(x) = f(x, y, z)
                I1 += wyj * wz[k] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
            end
        end

        # Y-axis
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k in eachindex(zs)
                z = zs[k]
                gy(y) = f(x, y, z)
                I2 += wxi * wz[k] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
            end
        end

        # Z-axis
        I3 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                gz(z) = f(x, y, z)
                I3 += wxi * wy[j] * (nth_derivative(gz, zL, dord) - nth_derivative(gz, zR, dord))
            end
        end

        return K * h^p * (I1 + I2 + I3)
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    I1 = 0.0
    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k in eachindex(zs)
            z = zs[k]
            gx(x) = f(x, y, z)
            I1 += wyj * wz[k] * nth_derivative(gx, x̄, m)
        end
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k in eachindex(zs)
            z = zs[k]
            gy(y) = f(x, y, z)
            I2 += wxi * wz[k] * nth_derivative(gy, ȳ, m)
        end
    end

    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            gz(z) = f(x, y, z)
            I3 += wxi * wy[j] * nth_derivative(gz, z̄, m)
        end
    end

    return C*(bb-aa)*h^m*(I1 + I2 + I3)
end

# ============================================================
# 4D derivative-based tensor-product error (leading terms)
# ============================================================

"""
    estimate_error_4d(
        f, 
        a::Real, 
        b::Real, 
        N::Int, 
        rule::Symbol
    ) -> Float64

Estimate a 4D integration error *scale* over the hypercube domain `[a,b]^4`
using a derivative-based tensor-product heuristic, with optional boundary-
difference handling for selected open-chain rules.

# Function description
This routine returns a **fast error scale model** intended for stabilizing
convergence fits / extrapolation, not a rigorous truncation bound.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules flagged by `_has_boundary_error_model(rule)`, the estimator applies
a boundary-difference leading-term model **axis by axis**.

For each axis, take a boundary difference of the corresponding axis-wise
derivative and integrate over the remaining three coordinates:

- X-axis:
  `I_x = ∭ [∂_x^{dord} f(xL, y, z, t) - ∂_x^{dord} f(xR, y, z, t)] dy dz dt`
- Y-axis:
  `I_y = ∭ [∂_y^{dord} f(x, yL, z, t) - ∂_y^{dord} f(x, yR, z, t)] dx dz dt`
- Z-axis:
  `I_z = ∭ [∂_z^{dord} f(x, y, zL, t) - ∂_z^{dord} f(x, y, zR, t)] dx dy dt`
- T-axis:
  `I_t = ∭ [∂_t^{dord} f(x, y, z, tL) - ∂_t^{dord} f(x, y, z, tR)] dx dy dz`

The final model is

`E ≈ K * h^p * (I_x + I_y + I_z + I_t)`,

where
- `h = (b-a)/N`,
- `xL = a + off*h`, `xR = a + (N-off)*h` (and similarly for `y`, `z`, `t`),
- `(p, K, dord, off) = _boundary_error_params(rule)`.

This boundary-difference structure is designed to reflect the leading behavior
of certain endpoint-free chained formulas and improve χ² stability.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, this routine follows the legacy midpoint tensor heuristic:

- Build 1D quadrature nodes/weights `(xs, wx)` for the selected `rule`.
- Approximate four axis-wise contributions by integrating the `m`-th derivative
  along one axis while fixing the remaining coordinates at quadrature nodes:
  - `I_x = ∭ ∂_x^m f(x̄, y, z, t) dy dz dt`
  - `I_y = ∭ ∂_y^m f(x, ȳ, z, t) dx dz dt`
  - `I_z = ∭ ∂_z^m f(x, y, z̄, t) dx dy dt`
  - `I_t = ∭ ∂_t^m f(x, y, z, t̄) dx dy dz`
- Return the scale model

`E ≈ C * (b-a) * h^m * (I_x + I_y + I_z + I_t)`,

where `(m, C) = _rule_params_for_tensor_error(rule)`.

# Arguments
- `f`: 4D integrand callable `f(x, y, z, t)` (function, closure, or callable struct).
- `a`, `b`: Hypercube domain bounds.
- `N`: Number of subdivisions per axis defining `h = (b-a)/N`.
- `rule`: Integration rule symbol.

# Returns
- `Float64`: heuristic signed error estimate (scale model). If `rule` is not
  recognized (and no boundary model is defined), returns `0.0`.

# Notes
- This estimator intentionally matches the loop structure and accumulation style
  of the original implementation for reproducibility.
- Rule-specific constraints on `N` (divisibility, minimum size, etc.) are:
  - enforced in `quadrature_1d_nodes_weights` for the default midpoint path, and
  - enforced explicitly in the boundary-model branch for supported open rules.
"""
function estimate_error_4d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol
)

    aa = float(a)
    bb = float(b)
    h  = (bb-aa)/N

    x̄ = (aa+bb)/2
    ȳ = (aa+bb)/2
    z̄ = (aa+bb)/2
    t̄ = (aa+bb)/2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

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
        yL = xL; yR = xR
        zL = xL; zR = xR
        tL = xL; tR = xR

        # X-axis
        I1 = 0.0
        @inbounds for j in eachindex(ys)
            y = ys[j]
            wyj = wy[j]
            for k in eachindex(zs)
                z = zs[k]
                wyj_wzk = wyj * wz[k]
                for l in eachindex(ts)
                    t = ts[l]
                    gx(x) = f(x, y, z, t)
                    I1 += wyj_wzk * wt[l] * (nth_derivative(gx, xL, dord) - nth_derivative(gx, xR, dord))
                end
            end
        end

        # Y-axis
        I2 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for k in eachindex(zs)
                z = zs[k]
                wxi_wzk = wxi * wz[k]
                for l in eachindex(ts)
                    t = ts[l]
                    gy(y) = f(x, y, z, t)
                    I2 += wxi_wzk * wt[l] * (nth_derivative(gy, yL, dord) - nth_derivative(gy, yR, dord))
                end
            end
        end

        # Z-axis
        I3 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                wxi_wyj = wxi * wy[j]
                for l in eachindex(ts)
                    t = ts[l]
                    gz(z) = f(x, y, z, t)
                    I3 += wxi_wyj * wt[l] * (nth_derivative(gz, zL, dord) - nth_derivative(gz, zR, dord))
                end
            end
        end

        # T-axis
        I4 = 0.0
        @inbounds for i in eachindex(xs)
            x = xs[i]
            wxi = wx[i]
            for j in eachindex(ys)
                y = ys[j]
                wxi_wyj = wxi * wy[j]
                for k in eachindex(zs)
                    z = zs[k]
                    gt(t) = f(x, y, z, t)
                    I4 += wxi_wyj * wz[k] * (nth_derivative(gt, tL, dord) - nth_derivative(gt, tR, dord))
                end
            end
        end

        return K * h^p * (I1 + I2 + I3 + I4)
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    I1 = 0.0
    @inbounds for j in eachindex(ys)
        y = ys[j]
        wyj = wy[j]
        for k in eachindex(zs)
            z = zs[k]
            wyj_wzk = wyj * wz[k]
            for l in eachindex(ts)
                t = ts[l]
                gx(x) = f(x, y, z, t)
                I1 += wyj_wzk * wt[l] * nth_derivative(gx, x̄, m)
            end
        end
    end

    I2 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for k in eachindex(zs)
            z = zs[k]
            wxi_wzk = wxi * wz[k]
            for l in eachindex(ts)
                t = ts[l]
                gy(y) = f(x, y, z, t)
                I2 += wxi_wzk * wt[l] * nth_derivative(gy, ȳ, m)
            end
        end
    end

    I3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for l in eachindex(ts)
                t = ts[l]
                gz(z) = f(x, y, z, t)
                I3 += wxi_wyj * wt[l] * nth_derivative(gz, z̄, m)
            end
        end
    end

    I4 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wxi = wx[i]
        for j in eachindex(ys)
            y = ys[j]
            wxi_wyj = wxi * wy[j]
            for k in eachindex(zs)
                z = zs[k]
                gt(t) = f(x, y, z, t)
                I4 += wxi_wyj * wz[k] * nth_derivative(gt, t̄, m)
            end
        end
    end

    return C*(bb-aa)*h^m*(I1 + I2 + I3 + I4)
end

# ============================================================
# Unified public API
# ============================================================

"""
    estimate_error(
        f, 
        a, 
        b, 
        N, 
        dim, 
        rule
    ) -> Float64

Unified interface for estimating integration error in 1–4 dimensions.

# Function description
Dispatches to the corresponding dimension-specific estimator:
- `dim == 1` → `estimate_error_1d`
- `dim == 2` → `estimate_error_2d`
- `dim == 3` → `estimate_error_3d`
- `dim == 4` → `estimate_error_4d`

# Arguments
- `f`: Integrand function (expects `dim` positional arguments).
- `a`, `b`: Bounds for each dimension (interpreted as scalar bounds for a hypercube `[a,b]^dim`).
- `N`: Number of subdivisions per axis (subject to rule constraints in 1D; higher-D estimators reuse the same rule nodes/weights).
- `dim`: Number of dimensions (`Int`).
- `rule`: Integration rule symbol.

# Returns
- A `Float64` error estimate. If `dim` is outside 1–4 or `rule` is not recognized
  by the selected estimator, returns `0.0`.
"""
function estimate_error(
    f, 
    a, 
    b, 
    N, 
    dim, 
    rule
)
    if dim == 1
        return estimate_error_1d(f, a, b, N, rule)
    elseif dim == 2
        return estimate_error_2d(f, a, b, N, rule)
    elseif dim == 3
        return estimate_error_3d(f, a, b, N, rule)
    elseif dim == 4
        return estimate_error_4d(f, a, b, N, rule)
    else
        return 0.0  # TODO: higher-dim error estimators
    end
end

end  # module ErrorEstimator