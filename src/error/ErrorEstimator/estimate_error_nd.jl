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

Return a fast `d`-dimensional quadrature error *scale* proxy for the hypercube integral, where ``d = \\texttt{dim}``,
```math
\\int\\limits_{[a,b]^d} d^d x \\; f(x_1,x_2,\\ldots,x_d)
```
using a lightweight derivative-based tensor-product heuristic.

This routine generalizes the dimension-specific error estimators, [`estimate_error_1d`](@ref)/[`estimate_error_2d`](@ref)/[`estimate_error_3d`](@ref)/[`estimate_error_4d`](@ref) into a single dimension-agnostic model, while
preserving the same loop ordering and accumulation structure.

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

## (A) Boundary-difference model (selected open-chain rules)
For rules where [`_has_boundary_error_model`](@ref)`(rule)` is true, a boundary-based
leading-term model is applied **axis by axis**.

Define boundary points
```math
- ``x_L = a + \\texttt{z} \\, h``\\,, \\quad ``x_R = a + ( N - \\texttt{z} ) \\, h``\\,,
```
where ``\\displaystyle{h = \\frac{b-a}{N}}`` and `(p, K, m, z) =`[`_boundary_error_params`](@ref)`(rule)`.

For each axis ``\\mu = 1, \\ldots , d``, construct
```math
I_\\mu = \\int\\limits_{[a,b]^{d-1}} d^{d-1} x \\; \\left[ \\partial_\\mu^{\\texttt{m}} f(\\ldots,x_L,\\ldots) - \\partial_\\mu^{\\texttt{m}} f(\\ldots,x_R,\\ldots) \\right]
```
and return
```math
E \\approx \\texttt{K} \\, h^\\texttt{p} \\, \\sum_\\mu I_\\mu \\,.
```

This boundary-difference structure models the dominant truncation behavior of
certain opened composite (endpoint-free) rule formulas and typically improves least ``\\chi^2`` fitting stability during ``h \\to 0`` extrapolation.

### Derivative evaluation and [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback
All derivatives in this routine are evaluated via the internal helper `_nth_deriv_safe`:
1) compute using [`nth_derivative`](@ref) ([`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)-based),
2) if non-finite, emit a [`Maranatha.JobLoggerTools.warn_benji`](@ref) and retry with [`nth_derivative_taylor`](@ref) ([`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl)-based),
3) throw an error only if the [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback is also non-finite.

Rule-specific constraints on `N` (divisibility, minimum size, etc.) are enforced
explicitly in this branch for supported opened composite rules.

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, the error estimator falls back to the midpoint tensor-product heuristic:

1) Obtain `(m, C)` via [`_rule_params_for_tensor_error`](@ref)`(rule)`.
2) Build `1`-dimensional nodes/weights `(xs, wx)` via [`Maranatha.Integrate.quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule)`.
3) For each axis ``\\mu``, integrate the `m`-th axis derivative over the remaining coordinates
   using tensor-product weights, by sampling ``\\partial_\\mu^{\\texttt{m}} f`` at the midpoint along that axis and
   enumerating quadrature nodes on the other axes (mirroring the legacy loop structure).
4) Return ``E \\approx C \\; (b-a) \\; h^{\\texttt{m}} \\;  \\sum_\\mu I_\\mu``.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Domain bounds defining the hypercube ``[a,b]^{d}``.
- `N`: Number of subdivisions per axis (``\\displaystyle{h = \\frac{b-a}{N}}``).
- `rule`: Integration rule symbol.
- `dim`: Number of spatial dimensions (must satisfy `dim```\\ge 1``).

# Returns
- `Float64`: A heuristic (signed) error scale proxy. If `m == 0` for the selected
  `rule`, returns `0.0`.

# Implementation notes
- The loop ordering, accumulation structure, and floating-point behavior intentionally
  mirror the dimension-specific implementations for reproducibility.
- The helper `_call_with_axis` dynamically replaces one coordinate while keeping the
  remaining axes fixed, enabling AD-compatible evaluation along the differentiated axis.
- Some rules may have **negative quadrature weights**; this estimator
  intentionally preserves the rule-defined weights rather than enforcing normalization.

# Limitations
- This estimator is not a rigorous error bound; it provides only a leading scaling model
  suitable for convergence fitting.
- Computational cost scales as ``O(d \\; \\texttt{length(xs)}^{\\texttt{dim}-1} )``, which grows rapidly with `dim`.
- The [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) fallback requires the integrand to accept generic number types
  (e.g. `Taylor1`). If the integrand dispatch is restricted to `Real` only, the fallback
  may raise a `MethodError`.

# Errors
- Throws an error if `dim < 1`.
- Throws an error if `(N, rule)` violates rule constraints.
- Throws an error if both [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`TaylorSeries.jl`](https://github.com/JuliaDiff/TaylorSeries.jl) derivatives are non-finite in the
  selected estimator branch.
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