# ============================================================================
# src/ErrorEstimate/ErrorGauss.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorGauss

import ..JobLoggerTools
import ..Quadrature

# ------------------------------------------------------------
# Float64 midpoint residual terms for composite GAUSS rules
# u ∈ [0, Nsub], center c = Nsub/2
# diff(k)  = ∫_0^N (u-c)^k du - Σ_i W[i]*(U[i]-c)^k
# coeff(k) = diff(k) / k!
# ------------------------------------------------------------

"""
    _exact_moment_shifted_float(
        Nsub::Int, 
        c::Float64, 
        k::Int
    ) -> Float64

Compute the exact shifted monomial moment
``\\displaystyle{ \\int\\limits_{0}^{N_{\\texttt{sub}}} du \\; (u - c)^k}``
in `Float64`.

# Function description
This helper returns the closed-form integral:
```math
\\int\\limits_{0}^{N} du \\; (u-c)^k
= \\left[ \\frac{(u-c)^{k+1}}{k+1} \\right]_{u=0}^{u=N}
= \\frac{(N-c)^{k+1} - (0-c)^{k+1}}{k+1} \\,,
```

with `N = Nsub` treated as a dimensionless length (unit-block tiling on ``u \\in [0, N_\\texttt{sub}]``).

It is used by [`_leading_midpoint_residual_terms_gauss_float`](@ref) to form the
exact-minus-quadrature residual for the midpoint-shifted monomial basis.

# Arguments

* `Nsub`: Number of composite unit blocks; the integration domain is ``u \\in [0, N_\\texttt{sub}]``.
* `c`: Shift (typically the midpoint), often ``c = \\dfrac{N_\\texttt{sub}}{2}``.
* `k`: Nonnegative integer power in ``\\left( u - c \\right)^k``.

# Returns

* `Float64`: The exact moment value in `Float64`.

# Notes

* This routine is *`Float64`-only* by design; it mirrors the numerical tolerance
  philosophy used by the Gauss residual-term generator.
* For large `k`, powers can overflow or lose accuracy in `Float64`; callers should
  keep `kmax` moderate.
"""
@inline function _exact_moment_shifted_float(
    Nsub::Int, 
    c::Float64, 
    k::Int
)::Float64
    Nf = Float64(Nsub)
    kp1 = Float64(k + 1)
    return ((Nf - c)^(k + 1) - (0.0 - c)^(k + 1)) / kp1
end

"""
    _leading_midpoint_residual_terms_gauss_float(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 2,
        kmax::Int = 128
    ) -> (ks, coeffs)

Detect the leading nonzero midpoint-shifted residual terms for a **composite Gauss-family**
quadrature rule on the dimensionless interval ``u \\in [0, N_\\texttt{sub}]`` (`Float64`-only).

# Function description

This routine numerically probes the residual moments of a composite Gauss rule by comparing:

* the exact shifted moment:

```math
M_k^{\\texttt{exact}} = \\int\\limits_{0}^{N} du \\; \\left( u - c \\right)^k \\,,
```

* against the quadrature approximation on the composite Gauss grid:

```math
M_k^{\\texttt{quad}}  = \\sum_i W_i \\, \\left( U_i - c \\right)^k \\,,
```

where ``(U, W)`` are produced by
[`Quadrature.Gauss._composite_gauss_u_grid`](@ref) and ``c = \\dfrac{N}{2}`` is the midpoint shift.

The difference is:

```math
\\texttt{diff}_k = M_k^{\\texttt{exact}} - M_k^{\\texttt{quad}}.
```

For each detected nonzero residual moment index ``k``, this routine records the
factorial-scaled coefficient:

```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}.
```

It returns the first `nterms` detected `(k, coeff(k))` pairs (as aligned vectors),
searching ``k = 0 , \\ldots , \\texttt{kmax}``.

# Arguments

* `rule`: Gauss rule symbol of the form `:gauss_pK` (`K` = points per block).
* `boundary`: Boundary-family selector, forwarded to Gauss:
  `:LU_EXEX` (Legendre), `:LU_INEX` (left Radau), `:LU_EXIN` (right Radau), `:LU_ININ` (Lobatto).
* `Nsub`: Number of unit blocks in the composite tiling (``u \\in [0, N_\\texttt{sub}]``).

# Keywords

* `nterms`: Number of leading nonzero residual terms to collect (must satisfy `nterms ≥ 1`).
* `kmax`: Maximum moment order to scan (must satisfy `kmax ≥ 0`).

# Returns

* `ks::Vector{Int}`: Moment indices where a nonzero residual was detected (length `nterms`).
* `coeffs::Vector{Float64}`: Factorial-scaled residual coefficients ``\\dfrac{\\texttt{diff}_k}{k!}`` aligned with `ks`.

# Error conditions

* Throws (via [`JobLoggerTools.error_benji`](@ref)) if:

  * `nterms < 1` or `kmax < 0`,
  * `rule` is not `:gauss_pK`,
  * the requested Gauss family constraints are violated (e.g., Radau/Lobatto need enough points),
  * or fewer than `nterms` nonzero residual moments are found up to `kmax`.

# Numerical tolerances

A residual is treated as nonzero if:
```math
\\left\\lvert \\texttt{diff}_k \\right\\rvert > 
\\texttt{tol\\_abs} + 
\\texttt{tol\\_rel} \\, \\left\\lvert M_k^{\\texttt{exact}} \\right\\rvert \\,,
```
where:

* `tol_abs = 5e4 * eps(Float64)`
* `tol_rel = 5e4 * eps(Float64)`

This is intentionally loose to avoid false positives from floating-point noise,
while still detecting genuine leading residual orders.

# Notes

* This routine is intended as a lightweight *order detection / coefficient extraction*
  tool for midpoint-based residual models; it is not a rigorous error bound.
* For large ``k``, ``\\left( U_i - c \\right)^k`` and ``\\left( N - c \\right)^{k+1}`` may overflow or underflow in `Float64`.
  Increase `kmax` cautiously.
"""
function _leading_midpoint_residual_terms_gauss_float(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 2,
    kmax::Int = 128
)::Tuple{Vector{Int}, Vector{Float64}}

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0")

    Quadrature.Gauss._is_gauss_rule(rule) || JobLoggerTools.error_benji("expected :gauss_pK (got $rule)")
    npts = Quadrature.Gauss._parse_gauss_p(rule)

    # dimensionless u-grid for composite Gauss on [0, Nsub]
    U, W = Quadrature.Gauss._composite_gauss_u_grid(Nsub, npts, boundary)

    c = Float64(Nsub) / 2.0

    # tolerances for Float64 (same spirit as your generator)
    tol_abs = 5e4 * eps(Float64)
    tol_rel = 5e4 * eps(Float64)

    ks = Int[]
    coeffs = Float64[]

    inv_fact = 1.0  # 1/0!
    for k in 0:kmax
        exact = _exact_moment_shifted_float(Nsub, c, k)

        approx = 0.0
        @inbounds for i in eachindex(U)
            approx += W[i] * (U[i] - c)^k
        end

        diff = exact - approx
        if abs(diff) > (tol_abs + tol_rel * abs(exact))
            push!(ks, k)
            push!(coeffs, diff * inv_fact)
            length(ks) == nterms && return ks, coeffs
        end

        # update inv_fact -> 1/k!
        if k >= 0
            kk = k + 1
            inv_fact /= kk
            inv_fact == 0.0 && break
        end
    end

    JobLoggerTools.error_benji("Could not collect nterms=$nterms GAUSS residual terms up to kmax=$kmax (Nsub=$Nsub).")
end

end  #  module ErrorGauss