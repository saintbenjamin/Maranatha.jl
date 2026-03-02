# ============================================================================
# src/generator/ErrorEstimate/error_estimate_2d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_2d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Estimate a ``2``-dimensional tensor-product truncation-error *model* for a composite Newton-Cotes rule
on the square domain ``[a,b]^2`` using the exact midpoint residual expansion.

# Function description
This routine extends the ``1``-dimensional midpoint-residual model axis-by-axis to a tensor-product setting.

Let ``\\displaystyle{h = \\frac{b-a}{N}}`` and ``\\displaystyle{\\bar{x} = \\bar{y} = \\frac{a+b}{2}}``.
From the exact ``1``-dimensional composite rule (via rational weight assembly), the midpoint residual expansion
yields a sequence of nonzero residual orders ``k`` with exact coefficients
``\\displaystyle{\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}}``. This routine collects the first `nerr_terms`
nonzero residual orders ``k_1, k_2, \\ldots`` (up to `kmax`) and returns the summed separable model:
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i}\\, h^{k_i+1}\\, \\left( I_x^{(k_i)} + I_y^{(k_i)} \\right),
```
with axis-wise cross integrals

```math
I_x^{(k)} = \\int\\limits_{a}^{b} dy\\; \\frac{\\partial^k f}{\\partial x^k}(\\bar{x}, y),
\\qquad
I_y^{(k)} = \\int\\limits_{a}^{b} dx\\; \\frac{\\partial^k f}{\\partial y^k}(x, \\bar{y}),
```
evaluated numerically using the same ``1``-dimensional composite nodes/weights.

# Arguments

* `f`:
  Scalar callable integrand ``f(x,y)`` (function, closure, or callable struct).
* `a`, `b`:
  Scalar bounds defining the square domain ``[a,b]^2``.
* `N`:
  Number of subintervals per axis. Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
* `boundary`:
  Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Keyword arguments

* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model (`1` = LO only, `2` = LO+NLO, ...).
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `Float64`:
  The summed separable truncation-error model value.

# Errors

* Propagates errors from:

  * composite weight assembly,
  * residual-term extraction,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
  insufficient nonzero residual terms exist up to `kmax`.

# Notes

* This model sums only *axis-separable* contributions (``x``-only and ``y``-only operators).
* Mixed derivative terms (e.g. ``\\partial_x^r\\partial_y^s f``) appear at higher asymptotic order
  and are intentionally omitted here.
* Coefficients are derived in exact rational arithmetic and converted to `Float64` only at the final stage.
"""
function error_estimate_2d(
    f, 
    a::Real, 
    b::Real, 
    N::Int, 
    rule::Symbol,
    boundary::Symbol;
    nerr_terms::Int = 1,
    kmax::Int = 128
)

    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2
    ȳ = (aa + bb) / 2

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    # collect LO / LO+NLO / ...
    ks, coeffsR = if nerr_terms == 1
        k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=min(kmax, 64))
        k == 0 && return 0.0
        ([k], Quadrature.RBig[coeffR])
    else
        _leading_midpoint_residual_terms(rule, boundary, N; nterms=nerr_terms, kmax=kmax)
    end

    err = 0.0

    @inbounds for it in eachindex(ks)
        k = ks[it]
        k == 0 && continue
        coeff = Float64(coeffsR[it])

        # X-axis contribution: apply k-th derivative in x, integrate over y
        I1 = 0.0
        for j in eachindex(xs)
            y = xs[j]
            gx(x) = f(x, y)
            I1 += wx[j] * nth_derivative(
                gx, x̄, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:x, stage=:midpoint
            )
        end

        # Y-axis contribution: apply k-th derivative in y, integrate over x
        I2 = 0.0
        for i in eachindex(xs)
            x = xs[i]
            gy(y) = f(x, y)
            I2 += wx[i] * nth_derivative(
                gy, ȳ, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:y, stage=:midpoint
            )
        end

        err += coeff * h^(k+1) * (I1 + I2)
    end

    return err
end

"""
    error_estimate_2d_threads(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Threaded variant of [`error_estimate_2d`](@ref) for 2D midpoint-residual truncation-error modeling.

All non-threading details (mathematical definition, coefficient construction, residual-term
interpretation, and overall intent) are identical to [`error_estimate_2d`](@ref).
See that function for the full formalism and background.

# Threading implementation

This function parallelizes the dominant axis-wise summation loops using Julia's built-in
multithreading:

* For each residual order `k` in `ks`, the ``x``-axis and ``y``-axis contributions are computed
  as weighted sums over the 1D quadrature nodes `xs`.
* Each axis sum is distributed via [`Base.Threads.@threads`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads) over the corresponding node loop.
* Each threaded node loop accumulates into a thread-local `Float64` scratch buffer
  indexed by [`Threads.threadid()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.threadid), followed by a `sum` reduction to form each axis integral.

Threading is enabled when Julia is started with `JULIA_NUM_THREADS > 1`.

# Arguments

Same as [`error_estimate_2d`](@ref).

# Keyword arguments

Same as [`error_estimate_2d`](@ref).

# Returns

Same as [`error_estimate_2d`](@ref).

# Notes

* This is an asymptotic *model* (fit stabilization / scaling diagnostics), not a strict bound.
* For small `length(xs)` or small `nerr_terms`, threading overhead may dominate.
"""
function error_estimate_2d_threads(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2
    ȳ = x̄

    xs, wx = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    ks, coeffsR = if nerr_terms == 1
        k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=min(kmax, 64))
        k == 0 && return 0.0
        ([k], Quadrature.RBig[coeffR])
    else
        _leading_midpoint_residual_terms(rule, boundary, N; nterms=nerr_terms, kmax=kmax)
    end

    L  = length(xs)
    nt = Threads.maxthreadid()

    err_total = 0.0

    @inbounds for it in eachindex(ks)
        k = ks[it]
        k == 0 && continue
        coeff = Float64(coeffsR[it])

        # -----------------------------
        # X-axis term (sum over y)
        # -----------------------------
        I1_parts = zeros(Float64, nt)

        Threads.@threads for j in 1:L
            tid = Threads.threadid()

            y   = xs[j]
            w   = wx[j]

            gx = let y=y
                x -> f(x, y)
            end

            dx = nth_derivative(
                gx, x̄, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:x, stage=:midpoint
            )

            I1_parts[tid] += w * dx
        end
        I1 = sum(I1_parts)

        # -----------------------------
        # Y-axis term (sum over x)
        # -----------------------------
        I2_parts = zeros(Float64, nt)

        Threads.@threads for i in 1:L
            tid = Threads.threadid()

            x   = xs[i]
            w   = wx[i]

            gy = let x=x
                y -> f(x, y)
            end

            dy = nth_derivative(
                gy, ȳ, k;
                h=h, rule=rule, N=N, dim=2,
                side=:mid, axis=:y, stage=:midpoint
            )

            I2_parts[tid] += w * dy
        end
        I2 = sum(I2_parts)

        err_total += coeff * h^(k + 1) * (I1 + I2)
    end

    return err_total
end