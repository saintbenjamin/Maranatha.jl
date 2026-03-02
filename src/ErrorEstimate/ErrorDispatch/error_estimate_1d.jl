# ============================================================================
# src/ErrorEstimate/ErrorDispatch/error_estimate_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_1d(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Estimate a ``1``-dimensional truncation-error *model* for a composite Newton-Cotes rule using
the exact midpoint residual expansion derived from rational weight assembly.

# Function description
This routine computes an asymptotic truncation-error model consistent with the
exact composite Newton-Cotes construction implemented in [`Maranatha.Quadrature`](@ref).

Let ``\\displaystyle{h = \\frac{b-a}{N}}`` and the physical midpoint ``\\displaystyle{\\bar{x} = \\frac{a+b}{2}}``.
Using exact rational composite weights ``\\beta`` (assembled internally), the midpoint-centered
residual moments determine a sequence of nonzero residual orders ``k`` and coefficients
``\\displaystyle{\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}}``, where:
```math
\\texttt{diff}_k
= \\int\\limits_0^{N} du\\,(u-c)^k
- \\sum_{j=0}^{N} \\beta_j\\,(j-c)^k,
\\qquad c = \\frac{N}{2}.
```

The function collects the first `nerr_terms` nonzero residual orders
``k_1, k_2, \\ldots`` (up to `kmax`) and returns the summed model:

```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i}\\, h^{k_i+1}\\, f^{(k_i)}(\\bar{x}).
```

# Arguments

* `f`:
  Scalar callable integrand ``f(x)`` (function, closure, or callable struct).
* `a`, `b`:
  Lower and upper bounds of the integration interval.
* `N`:
  Number of subintervals.
  Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
* `boundary`:
  Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Keyword arguments

* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model.
  `1` gives LO only; `2` gives LO+NLO; etc.
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `Float64`:
  The summed truncation-error model value (may be signed, following the model coefficients
  and derivative signs).

# Errors

* Propagates errors from:

  * composite weight assembly,
  * residual-term extraction,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.Utils.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
  insufficient nonzero residual terms exist up to `kmax`.

# Notes

* This is an asymptotic *model* for fit stabilization and scaling diagnostics, not a strict bound.
* Coefficients are derived in exact rational arithmetic and converted to `Float64` only at the final stage.
"""
function error_estimate_1d(
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

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2

    # # Collect LO, (optionally) NLO, ... midpoint residual terms
    # ks, coeffsR = _leading_midpoint_residual_terms(
    #     rule, boundary, N;
    #     nterms = nerr_terms,
    #     kmax   = kmax
    # )
    ks, coeffs, _center = _leading_residual_terms_any(rule, boundary, N; nterms=nerr_terms, kmax=kmax)

    err = 0.0

    @inbounds for i in eachindex(ks)
        k = ks[i]
        k == 0 && continue  # degenerate safety
        # coeff = Float64(coeffsR[i])
        coeff = coeffs[i]

        dx = nth_derivative(
            f,
            x̄, k;
            h=h, rule=rule, N=N, dim=1,
            side=:mid, axis=:x, stage=:midpoint
        )

        err += coeff * h^(k+1) * dx
    end

    return err
end

"""
    error_estimate_1d_threads(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Threaded variant of [`error_estimate_1d`](@ref) for 1D midpoint-residual truncation-error modeling.

All non-threading details (mathematical definition, coefficient construction, residual-term
interpretation, and overall intent) are identical to [`error_estimate_1d`](@ref).
See that function for the full formalism and background.

# Threading implementation

This function parallelizes the loop over collected residual orders `ks` using Julia's
built-in multithreading:

* The work is distributed via [`Base.Threads.@threads`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads) over `eachindex(ks)`.
* Each iteration computes one derivative term ``f^{(k)}(\\bar{x})`` and its contribution to ``E``.
* Each thread accumulates its contribution into a thread-local `Float64`
  buffer indexed by [`Threads.threadid()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.threadid).
* After the threaded loop completes, the per-thread partial sums are
  combined via a final `sum` reduction.

Threading is enabled when Julia is started with `JULIA_NUM_THREADS > 1`.

# Arguments

Same as [`error_estimate_1d`](@ref).

# Keyword arguments

Same as [`error_estimate_1d`](@ref).

# Returns

Same as [`error_estimate_1d`](@ref).

# Notes

* This is an asymptotic *model* (fit stabilization / scaling diagnostics), not a strict bound.
* For small `length(ks)` (common when `nerr_terms` is small), threading overhead may dominate.
"""
function error_estimate_1d_threads(
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

    # ks, coeffsR = _leading_midpoint_residual_terms(
    #     rule, boundary, N;
    #     nterms = nerr_terms,
    #     kmax   = kmax
    # )
    ks, coeffs, _center = _leading_residual_terms_any(rule, boundary, N; nterms=nerr_terms, kmax=kmax)

    nt = Threads.maxthreadid()

    # Each thread accumulates into its own slot; no atomics.
    parts = zeros(Float64, nt)

    @inbounds Threads.@threads for i in eachindex(ks)
        tid = Threads.threadid()

        k = ks[i]
        k == 0 && return
        # coeff = Float64(coeffsR[i])
        coeff = coeffs[i]

        dx = nth_derivative(
            f, x̄, k;
            h=h, rule=rule, N=N, dim=1,
            side=:mid, axis=:x, stage=:midpoint
        )

        parts[tid] += coeff * h^(k + 1) * dx
    end

    return sum(parts)
end