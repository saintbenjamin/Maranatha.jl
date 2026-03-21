# ============================================================================
# src/ErrorEstimate/ErrorGaussDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ErrorGaussDerivative

Residual-model backend for derivative-based error estimation with Gauss-family
quadrature rules.

# Module description
`ErrorGaussDerivative` implements the rule-family-specific residual analysis
used by derivative-based error estimators for Gauss, Gauss-Radau, and
Gauss-Lobatto composite quadrature rules.

It provides helpers for:

- identifying leading midpoint residual terms,
- converting those residuals into factorial-scaled coefficients,
- exposing Gauss-family residual models to the generic derivative dispatch
  layer.

# Notes
- This is an internal module.
- Higher-level orchestration is performed by `ErrorDispatchDerivative`.
"""
module ErrorGaussDerivative

import ..JobLoggerTools
import ..Gauss

# ------------------------------------------------------------
# Float64 midpoint residual terms for composite Gauss rules
# u ∈ [0, Nsub], center c = Nsub/2
# diff(k)  = ∫_0^N (u-c)^k du - Σ_i W[i]*(U[i]-c)^k
# coeff(k) = diff(k) / k!
# ------------------------------------------------------------

"""
    _exact_moment_shifted_float(
        Nsub::Int, 
        c, 
        k::Int
    ) -> Real

Compute the exact shifted monomial moment
``\\displaystyle{ \\int\\limits_{0}^{N_{\\texttt{sub}}} du \\; (u - c)^k}``
in `Float64`.

# Function description
This helper returns the closed-form shifted moment
```math
\\int\\limits_0^N (u-c)^k \\, du
=
\\frac{(N-c)^{k+1} - (0-c)^{k+1}}{k+1},
```
with `N = Nsub` interpreted as the dimensionless length of the composite
unit-block tiling.

It is used by [`_leading_midpoint_residual_terms_gauss_float`](@ref) to compare
the exact midpoint-shifted monomial moment against the quadrature-induced one.

# Arguments
- `Nsub`: Number of composite unit blocks, so the domain is ``u \\in [0, N_{\\texttt{sub}}]``.
- `c`: Shift value, typically the midpoint ``\\dfrac{N_{\\texttt{sub}}}{2}``.
- `k`: Nonnegative integer power.

# Returns
- `Real`:
  Exact shifted moment value in the same scalar type as `c`.

# Errors
- No explicit validation is performed here; invalid inputs are assumed to be
  filtered by callers.

# Notes
- Despite the historical function name, this helper follows the scalar type of
  `c` and is not restricted to `Float64`.
- It is used by [`_leading_midpoint_residual_terms_gauss_float`](@ref) to compare
  the exact midpoint-shifted monomial moment against the quadrature-induced one.
- Large `k` can lead to overflow or loss of accuracy depending on the active scalar type.
"""
@inline function _exact_moment_shifted_float(
    Nsub::Int,
    c,
    k::Int
)
    T = typeof(c)
    Nf = T(Nsub)
    kp1 = T(k + 1)
    return ((Nf - c)^(k + 1) - (zero(T) - c)^(k + 1)) / kp1
end

"""
    _leading_midpoint_residual_terms_gauss_float(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 2,
        kmax::Int = 128,
        real_type = Float64,
    ) -> Tuple

Detect the leading nonzero midpoint-shifted residual terms for a composite Gauss-family rule.

# Function description
This routine probes the midpoint-centered residual structure of a composite
Gauss-family quadrature rule on the dimensionless interval
``u \\in [0, N_{\\texttt{sub}}]``.

It compares, for each scanned order ``k``,

```math
M_k^{\\texttt{exact}} = \\int\\limits_0^N du \\; (u-c)^k
```

against

```math
M_k^{\\texttt{quad}} = \\sum_i W_i (U_i-c)^k,
```

where `(U, W)` is the composite Gauss grid returned by
[`Gauss._composite_gauss_u_grid`](@ref) and
``c = \\dfrac{N}{2}`` is the midpoint.

The residual
```math
\\texttt{diff}_k = M_k^{\\texttt{exact}} - M_k^{\\texttt{quad}}
```
is converted into the factorial-scaled coefficient
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}.
```

The function returns the first `nterms` detected nonzero residual terms as
aligned vectors `(ks, coeffs)`.

# Arguments
- `rule`: Gauss-family rule symbol of the form `:gauss_p2`, `:gauss_p3`, etc.
- `boundary`: Boundary-family selector passed to the Gauss backend.
- `Nsub`: Number of composite unit blocks on the dimensionless grid.

# Keyword arguments
- `nterms`: Number of leading nonzero residual terms to collect.
- `kmax`: Maximum moment order to scan.
- `real_type = Float64`:
  Scalar type used internally for Gauss grid construction, moment evaluation,
  and residual coefficients.

# Returns
- `ks::Vector{Int}`: 
  Residual orders where a nonzero moment is detected.
- `coeffs`:
  Factorial-scaled coefficients aligned with `ks`, stored in the active
  `real_type`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1` or `kmax < 0`.
- Throws if `rule` is not of the form `:gauss_pK`.
- Propagates backend errors from the composite Gauss grid construction.
- Throws if fewer than `nterms` nonzero residual moments are found up to `kmax`.

# Notes
- Residual detection is tolerance-based rather than exact.
- This routine is intended for leading-order / coefficient extraction, not for
  rigorous error bounds.
- Despite the historical function name, this routine supports configurable
  scalar types through `real_type`.
"""
function _leading_midpoint_residual_terms_gauss_float(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 2,
    kmax::Int = 128,
    real_type = Float64,
)::Tuple

    T = real_type

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0")

    Gauss._is_gauss_rule(rule) || JobLoggerTools.error_benji("expected :gauss_pK (got $rule)")
    npts = Gauss._parse_gauss_p(rule)

    U, W = Gauss._composite_gauss_u_grid(Nsub, npts, boundary; real_type = T)

    c = T(Nsub) / T(2)

    tol_abs = T(5e4) * eps(T)
    tol_rel = T(5e4) * eps(T)

    ks = Int[]
    coeffs = T[]

    inv_fact = one(T)
    for k in 0:kmax
        exact = _exact_moment_shifted_float(Nsub, c, k)

        approx = zero(T)
        @inbounds for i in eachindex(U)
            approx += W[i] * (U[i] - c)^k
        end

        diff = exact - approx
        if abs(diff) > (tol_abs + tol_rel * abs(exact))
            push!(ks, k)
            push!(coeffs, diff * inv_fact)
            length(ks) == nterms && return ks, coeffs
        end

        kk = k + 1
        inv_fact /= T(kk)
        inv_fact == zero(T) && break
    end

    JobLoggerTools.error_benji("Could not collect nterms=$nterms Gauss residual terms up to kmax=$kmax (Nsub=$Nsub).")
end

end  #  module ErrorGaussDerivative
