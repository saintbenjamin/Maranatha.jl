# ============================================================================
# src/error/ErrorEstimator/estimate_error_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_nd(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol;
        dim::Int
    ) -> Float64

Return a fast **nD integration error scale** proxy for the hypercube integral
`∫_{[a,b]^dim} f(x₁, …, x_dim) d^dim x` over `[a,b]^dim`, using a lightweight
derivative-based tensor-product heuristic.

This routine generalizes the dimension-specific estimators
`estimate_error_1d/2d/3d/4d` into a single dimension-agnostic model, while
preserving the same loop ordering and accumulation structure.

For selected endpoint-free (“open-chain”) rules, the estimator switches to an
axis-wise boundary-difference proxy that is often more stable in χ²-based
convergence fits. All derivatives are attempted with ForwardDiff first, with an
automatic Taylor fallback when the ForwardDiff result is non-finite (`Inf`/`NaN`).

# Function description
This routine provides a **cheap, consistent error scale model** intended to:
- supply per-point weights `σ(h)` for χ²-based convergence fits (`fit_convergence`), and
- keep a consistent `h`-scaling proxy across 1D–nD estimators in this package.

It is **not** a rigorous truncation bound and does not attempt to reproduce the
full composite-rule error expansion.

Two regimes are supported:

## (A) Boundary-difference model (selected open-chain rules)
For rules where `_has_boundary_error_model(rule)` is true, a boundary-based
leading-term model is applied **axis by axis**.

Define boundary points

`xL = a + off*h`,  `xR = a + (N-off)*h`,

where `h = (b-a)/N` and `(p, K, dord, off) = _boundary_error_params(rule)`.

For each axis `μ = 1..dim`, construct

`I_μ = ∫_{[a,b]^{dim-1}} [ ∂_μ^{dord} f(…, xL, …) - ∂_μ^{dord} f(…, xR, …) ] d(other axes)`,

and return

`E ≈ K * h^p * Σ_μ I_μ`.

This boundary-difference structure models the dominant truncation behavior of
certain endpoint-free chained Newton–Cotes formulas and typically improves χ²
stability during extrapolation.

### Derivative evaluation and Taylor fallback
All derivatives are evaluated via the internal helper `_nth_deriv_safe`:
1) compute using `nth_derivative` (ForwardDiff-based),
2) if non-finite, emit a `warn_benji` and retry with `nth_derivative_taylor` (TaylorSeries-based),
3) throw an error only if the Taylor fallback is also non-finite.

Rule-specific constraints on `N` (divisibility, minimum size, etc.) are enforced
explicitly in this branch for supported open rules.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, the estimator falls back to the midpoint tensor-product heuristic:

1) Obtain `(m, C)` via `_rule_params_for_tensor_error(rule)`.
2) Build 1D quadrature nodes/weights `(xs, ws)` via `quadrature_1d_nodes_weights(a, b, N, rule)`.
3) For each axis `μ`, integrate the `m`-th axis derivative over the remaining coordinates
   using tensor-product weights, by sampling `∂_μ^m` at the midpoint along that axis and
   enumerating quadrature nodes on the other axes (mirroring the legacy loop structure).
4) Return

`E ≈ C * (b-a) * h^m * Σ_μ I_μ`.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Domain bounds defining the hypercube `[a,b]^dim`.
- `N`: Number of subdivisions per axis (`h = (b-a)/N`).
- `rule`: Integration rule symbol.
- `dim`: Number of spatial dimensions (must satisfy `dim ≥ 1`).

# Returns
- `Float64`: A heuristic (signed) error scale proxy. If `m == 0` for the selected
  `rule`, returns `0.0`.

# Implementation notes
- The loop ordering, accumulation structure, and floating-point behavior intentionally
  mirror the dimension-specific implementations for reproducibility.
- The helper `_call_with_axis` dynamically replaces one coordinate while keeping the
  remaining axes fixed, enabling AD-compatible evaluation along the differentiated axis.
- Some open-chain rules may have **negative quadrature weights**; this estimator
  intentionally preserves the rule-defined weights rather than enforcing normalization.

# Limitations
- This estimator is not a rigorous error bound; it provides only a leading scaling model
  suitable for convergence fitting.
- Computational cost scales as `O(dim * length(xs)^(dim-1))`, which grows rapidly with `dim`.
- The Taylor fallback requires that the integrand supports generic number types (e.g. `Taylor1`).
  If the integrand dispatch is restricted to `Real` only, the fallback may raise a `MethodError`.

# Errors
- Throws an error if `dim < 1`.
- Throws an error if `(N, rule)` violates rule constraints.
- Throws an error if both ForwardDiff and Taylor derivatives are non-finite in the selected
  estimator branch.
"""
function estimate_error_nd(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol;
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xs, ws = quadrature_1d_nodes_weights(aa, bb, N, rule)

    @inline function _nth_deriv_safe(g, x, n; side::Symbol=:mid, axis::Int=0, stage::Symbol=:mid)
        d = nth_derivative(g, x, n)
        if !isfinite(d)
            JobLoggerTools.warn_benji(
                "Non-finite derivative (ForwardDiff); trying Taylor fallback " *
                "h=$h x=$x n=$n rule=$rule N=$N side=$side axis=$axis stage=$stage dim=$dim"
            )
            d = nth_derivative_taylor(g, x, n)
            if !isfinite(d)
                JobLoggerTools.error_benji(
                    "Non-finite in nD error estimator even after Taylor fallback: " *
                    "h=$h x=$x deriv=$d n=$n rule=$rule N=$N side=$side axis=$axis stage=$stage dim=$dim"
                )
            end
        end
        return d
    end

    # helper: call f with axis value replaced by x (x may be Dual)
    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
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

        total_axes = 0.0

        fixed = Vector{Float64}(undef, dim)      # stores "other axes" values
        idx   = ones(Int, dim - 1)               # odometer for other axes

        @inbounds for axis in 1:dim
            Iaxis = 0.0

            if dim == 1
                # Iaxis = nth_derivative(x -> f(x), xL, dord) -
                #         nth_derivative(x -> f(x), xR, dord)
                Iaxis =
                    _nth_deriv_safe(x -> f(x), xL, dord; side=:L, axis=axis, stage=:boundary) -
                    _nth_deriv_safe(x -> f(x), xR, dord; side=:R, axis=axis, stage=:boundary)
            else
                fill!(idx, 1)
                while true
                    wprod = 1.0
                    t = 1
                    for d in 1:dim
                        if d == axis
                            continue
                        end
                        i = idx[t]
                        fixed[d] = xs[i]
                        wprod *= ws[i]
                        t += 1
                    end

                    # Iaxis += wprod * (
                    #     nth_derivative(x -> _call_with_axis(f, fixed, axis, x, dim), xL, dord) -
                    #     nth_derivative(x -> _call_with_axis(f, fixed, axis, x, dim), xR, dord)
                    # )
                    Iaxis += wprod * (
                        _nth_deriv_safe(x -> _call_with_axis(f, fixed, axis, x, dim), xL, dord; side=:L, axis=axis, stage=:boundary) -
                        _nth_deriv_safe(x -> _call_with_axis(f, fixed, axis, x, dim), xR, dord; side=:R, axis=axis, stage=:boundary)
                    )

                    # increment odometer on idx
                    q = dim - 1
                    while q >= 1
                        idx[q] += 1
                        if idx[q] <= length(xs)
                            break
                        else
                            idx[q] = 1
                            q -= 1
                        end
                    end
                    q == 0 && break
                end
            end

            total_axes += Iaxis
        end

        return K * h^p * total_axes
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xmid = (aa + bb) / 2

    total_axes = 0.0

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, dim - 1)

    @inbounds for axis in 1:dim
        Iaxis = 0.0

        if dim == 1
            # Iaxis = nth_derivative(x -> f(x), xmid, m)
            Iaxis = _nth_deriv_safe(x -> f(x), xmid, m; side=:mid, axis=axis, stage=:midpoint)
        else
            fill!(idx, 1)
            while true
                wprod = 1.0
                t = 1
                for d in 1:dim
                    if d == axis
                        continue
                    end
                    i = idx[t]
                    fixed[d] = xs[i]
                    wprod *= ws[i]
                    t += 1
                end

                # Iaxis += wprod * nth_derivative(
                #     x -> _call_with_axis(f, fixed, axis, x, dim),
                #     xmid, m
                # )
                Iaxis += wprod * _nth_deriv_safe(
                    x -> _call_with_axis(f, fixed, axis, x, dim),
                    xmid, m;
                    side=:mid, axis=axis, stage=:midpoint
                )

                # increment odometer on idx
                q = dim - 1
                while q >= 1
                    idx[q] += 1
                    if idx[q] <= length(xs)
                        break
                    else
                        idx[q] = 1
                        q -= 1
                    end
                end
                q == 0 && break
            end
        end

        total_axes += Iaxis
    end

    return C * (bb - aa) * h^m * total_axes
end