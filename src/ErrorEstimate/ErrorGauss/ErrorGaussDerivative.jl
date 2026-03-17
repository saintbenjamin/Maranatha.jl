# ============================================================================
# src/ErrorEstimate/ErrorGaussDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
        c::Float64, 
        k::Int
    ) -> Float64

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
- `Float64`: Exact shifted moment value in `Float64`.

# Errors
- No explicit validation is performed here; invalid inputs are assumed to be
  filtered by callers.

# Notes
- This helper is intentionally `Float64`-only and follows the tolerance-based
  philosophy of the Gauss residual backend.
- Large `k` can lead to overflow or loss of accuracy in `Float64`.
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

# Returns
- `ks::Vector{Int}`: Residual orders where a nonzero moment is detected.
- `coeffs::Vector{Float64}`: Factorial-scaled coefficients aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1` or `kmax < 0`.
- Throws if `rule` is not of the form `:gauss_pK`.
- Propagates backend errors from the composite Gauss grid construction.
- Throws if fewer than `nterms` nonzero residual moments are found up to `kmax`.

# Notes
- Residual detection is tolerance-based rather than exact.
- This routine is intended for leading-order / coefficient extraction, not for
  rigorous error bounds.
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

    Gauss._is_gauss_rule(rule) || JobLoggerTools.error_benji("expected :gauss_pK (got $rule)")
    npts = Gauss._parse_gauss_p(rule)

    # dimensionless u-grid for composite Gauss on [0, Nsub]
    U, W = Gauss._composite_gauss_u_grid(Nsub, npts, boundary)

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

    JobLoggerTools.error_benji("Could not collect nterms=$nterms Gauss residual terms up to kmax=$kmax (Nsub=$Nsub).")
end

end  #  module ErrorGaussDerivative