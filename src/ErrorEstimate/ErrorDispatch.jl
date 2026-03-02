# ============================================================================
# src/ErrorEstimate/ErrorDispatch.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorDispatch

using Base.Threads
import ..LinearAlgebra
import ..TaylorSeries
import ..Enzyme
import ..ForwardDiff

import ..JobLoggerTools

import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..Quadrature.QuadratureDispatch

import ..ErrorEstimate.ErrorNewtonCotes
import ..ErrorEstimate.ErrorGauss
import ..ErrorEstimate.ErrorBSpline

include("ErrorDispatch/nth_derivative.jl")

# ------------------------------------------------------------
# Unified residual-term collector (NS exact OR GAUSS float)
# returns (ks, coeffs_float, center_symbol)
# center is currently always :mid
# ------------------------------------------------------------

"""
    _leading_residual_terms_any(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 1,
        kmax::Int = 128
    ) -> (ks, coeffs_float, center)

Collect the first `nterms` nonzero **midpoint-shifted residual coefficients**
for either an NS (exact-rational) composite Newton–Cotes rule or a Float64 composite Gauss-family rule.

# Function description
This is a unified internal helper that normalizes the two residual backends into a common return type:

- **NS rules** (`:ns_pK`):
  Uses exact rational moment-matching residual coefficients computed from the exact-assembled
  composite Newton–Cotes coefficients `β`. The resulting residual coefficients are exact rationals
  internally, then converted to `Float64` in this wrapper.

- **GAUSS rules** (`:gauss_pK`):
  Uses Float64 probing of midpoint-shifted monomial moments on the composite dimensionless grid
  `u ∈ [0, Nsub]` (via `Gauss._composite_gauss_u_grid`) and identifies nonzero residual moments
  using tolerance-based criteria (see the Gauss backend implementation).

The residual is defined in terms of the midpoint shift `c` (currently always the midpoint):
```math
\\mathrm{diff}(k) = \\int_{0}^{N} (u-c)^k\\,du - \\sum_i W_i\\,(U_i-c)^k,
\\qquad
\\mathrm{coeff}(k)=\\frac{\\mathrm{diff}(k)}{k!}.
```

This routine returns the first `nterms` detected `(k, coeff(k))` pairs as aligned vectors.

# Arguments

* `rule`: Quadrature rule symbol. Supported:

  * `:ns_pK` (exact composite Newton–Cotes; handled by `NewtonCotes` + `ErrorNewtonCotes`)
  * `:gauss_pK` (composite Gauss family; handled by `Gauss` + `ErrorGauss`)
* `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
  This is validated by `QuadratureDispatch._decode_boundary(boundary)`.
* `Nsub`: Number of unit blocks in the dimensionless tiling domain `u ∈ [0, Nsub]`.

# Keyword arguments

* `nterms`: Number of leading nonzero residual terms to return (`nterms ≥ 1`).
* `kmax`: Maximum moment order to scan in the backend search.

# Returns

* `ks::Vector{Int}`:
  Residual moment indices `k` where a nonzero residual was detected (length `nterms`).
* `coeffs_float::Vector{Float64}`:
  Factorial-scaled residual coefficients `diff(k)/k!`, returned as `Float64` (length `nterms`).

  * For NS rules, these originate as exact rationals and are converted to Float64 here.
  * For GAUSS rules, these are produced directly in Float64.
* `center::Symbol`:
  Centering convention symbol. Currently always `:mid`.

# Errors

* Throws (via `JobLoggerTools.error_benji`) if:

  * `boundary` is invalid,
  * `rule` is neither NS nor GAUSS,
  * or the backend fails to collect the requested number of terms within `kmax`.

# Notes

* This is an internal building block for residual-based error-scale models; it is not part of
  the public API.
* The center is currently fixed to `:mid`, but the return includes it so future extensions can
  support alternative centers without changing downstream signatures.
"""
function _leading_residual_terms_any(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 1,
    kmax::Int = 128
)::Tuple{Vector{Int}, Vector{Float64}, Symbol}

    QuadratureDispatch._decode_boundary(boundary)

    if NewtonCotes._is_ns_rule(rule)
        # exact rational coefficients from β
        if nterms == 1
            k, coeffR = ErrorNewtonCotes._leading_midpoint_residual_term(rule, boundary, Nsub; kmax=min(kmax, 64))
            return [k], [Float64(coeffR)], :mid
        else
            ks, coeffsR = ErrorNewtonCotes._leading_midpoint_residual_terms(rule, boundary, Nsub; nterms=nterms, kmax=kmax)
            return ks, Float64.(coeffsR), :mid
        end
    end

    if Gauss._is_gauss_rule(rule)
        ks, coeffs = ErrorGauss._leading_midpoint_residual_terms_gauss_float(rule, boundary, Nsub; nterms=nterms, kmax=kmax)
        return ks, coeffs, :mid
    end

    if BSpline._is_bspl_rule(rule)
        ks, coeffs = ErrorBSpline._leading_midpoint_residual_terms_bspline_float(
            rule, boundary, Nsub; nterms=nterms, kmax=kmax, λ=0.0
        )
        return ks, coeffs, :mid
    end

    JobLoggerTools.error_benji("Unsupported rule for residual model: rule=$rule")
end

"""
    _leading_residual_ks_with_center_any(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int,
        kmax::Int = 256,
        tol_abs::Float64 = 5e4 * eps(Float64),
        tol_rel::Float64 = 5e4 * eps(Float64)
    ) -> (ks, center)

Extract only the indices `k` of the first `nterms` nonzero midpoint-shifted residual moments,
together with the centering convention.

# Function description
This is a lightweight variant of [`_leading_residual_terms_any`](@ref) that returns only the
moment indices `k` where a nonzero residual is detected, plus the center symbol.

Two backends are supported:

## (A) NS rules (`:ns_pK`) — exact rational detection
For NS composite Newton–Cotes rules, this routine assembles the exact rational composite
coefficient vector `β` and tests exact nonzero-ness:
```math
\\mathrm{diff}(k)=\\int_{0}^{N}(u-c)^k\\,du-\\sum_{j=0}^{N}\\beta_j\\,(j-c)^k,
\\quad \\text{(exact rational)}
```

A moment index `k` is recorded when `diff(k) != 0` in exact arithmetic.

## (B) GAUSS rules (`:gauss_pK`) — Float64 tolerance detection

For composite Gauss-family rules, this routine constructs the dimensionless grid
`(U, W)` on `u ∈ [0, Nsub]` and tests nonzero residual with a tolerance condition:

```math
|\\mathrm{diff}(k)| > \\texttt{tol_abs} + \\texttt{tol_rel}\\,|\\mathrm{exact}|.
```

The center is currently fixed as:

```math
c = Nsub/2
```

and returned as `:mid`.

# Arguments

* `rule`: Quadrature rule symbol. Supported:

  * `:ns_pK` (exact composite Newton–Cotes)
  * `:gauss_pK` (composite Gauss family)
* `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
* `Nsub`: Number of unit blocks in the dimensionless domain `u ∈ [0, Nsub]`.

# Keyword arguments

* `nterms`: Number of leading residual indices to return (must satisfy `nterms ≥ 1`).
* `kmax`: Maximum moment order to scan (must satisfy `kmax ≥ 0`).
* `tol_abs`: Absolute tolerance for GAUSS nonzero detection (Float64-only path).
* `tol_rel`: Relative tolerance for GAUSS nonzero detection (Float64-only path).

# Returns

* `ks::Vector{Int}`:
  Residual moment indices where nonzero residual was detected (length `nterms`).
* `center::Symbol`:
  Centering convention symbol (currently always `:mid`).

# Errors

* Throws (via `JobLoggerTools.error_benji`) if:

  * `nterms < 1` or `kmax < 0`,
  * `boundary` is invalid,
  * `rule` is unsupported,
  * or fewer than `nterms` residual indices are detected up to `kmax`.

# Notes

* This is intended for quickly selecting convergence powers / error-model orders without
  paying the cost of also collecting coefficient magnitudes.
* For GAUSS rules, the numerical decision boundary is controlled by `tol_abs`/`tol_rel`.
  For NS rules, the decision is exact (`diff != 0`).
"""
function _leading_residual_ks_with_center_any(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int,
    kmax::Int = 256,
    tol_abs::Float64 = 5e4 * eps(Float64),
    tol_rel::Float64 = 5e4 * eps(Float64)
)::Tuple{Vector{Int}, Symbol}

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1 (got $nterms)")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0 (got $kmax)")

    QuadratureDispatch._decode_boundary(boundary)
    center = :mid

    # ------------------------------------------------------------
    # NS rules: exact rational test (diff != 0)
    # ------------------------------------------------------------
    if NewtonCotes._is_ns_rule(rule)
        ks = Int[]
        c  = NewtonCotes.RBig(BigInt(Nsub), 2)
        Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)

        β = NewtonCotes._assemble_composite_beta_rational(NewtonCotes._parse_ns_p(rule), boundary, Nsub)

        for k in 0:kmax
            exact = ((Nrb - c)^(k+1) - (NewtonCotes.RBig(0) - c)^(k+1)) /
                    NewtonCotes.RBig(BigInt(k+1), 1)

            approx = NewtonCotes.RBig(0)
            @inbounds for j in 0:Nsub
                wj = β[j+1]; wj == 0 && continue
                approx += wj * (NewtonCotes.RBig(BigInt(j),1) - c)^k
            end

            if exact - approx != 0
                push!(ks, k)
                length(ks) == nterms && return ks, center
            end
        end

        JobLoggerTools.error_benji("Could not collect nterms=$nterms NS residual ks up to kmax=$kmax")
    end

    # ------------------------------------------------------------
    # GAUSS rules: Float64 test with tolerances
    # ------------------------------------------------------------
    if Gauss._is_gauss_rule(rule)
        npts = Gauss._parse_gauss_p(rule)

        # dimensionless u-grid on [0, Nsub]
        U, W = Gauss._composite_gauss_u_grid(Nsub, npts, boundary)
        c = Float64(Nsub) / 2.0

        ks = Int[]
        for k in 0:kmax
            # exact moment ∫_0^N (u-c)^k du  (Float64)
            exact = ((Float64(Nsub) - c)^(k+1) - (0.0 - c)^(k+1)) / Float64(k+1)

            approx = 0.0
            @inbounds for i in eachindex(U)
                approx += W[i] * (U[i] - c)^k
            end

            diff = exact - approx

            # IMPORTANT: tolerance-based "nonzero"
            if abs(diff) > (tol_abs + tol_rel * abs(exact))
                push!(ks, k)
                length(ks) == nterms && return ks, center
            end
        end

        JobLoggerTools.error_benji("Could not collect nterms=$nterms GAUSS residual ks up to kmax=$kmax")
    end

    if BSpline._is_bspl_rule(rule)
        ks, center = ErrorBSpline._leading_residual_ks_with_center_bspline_float(
            rule, boundary, Nsub; nterms=nterms, kmax=kmax, λ=0.0, tol_abs=tol_abs, tol_rel=tol_rel
        )
        return ks, center
    end

    JobLoggerTools.error_benji("Unsupported rule=$rule for residual ks extraction.")
end

include("ErrorDispatch/error_estimate_1d.jl")
include("ErrorDispatch/error_estimate_2d.jl")
include("ErrorDispatch/error_estimate_3d.jl")
include("ErrorDispatch/error_estimate_4d.jl")
include("ErrorDispatch/error_estimate_nd.jl")

# ============================================================
# Unified public API
# ============================================================

"""
    error_estimate(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        nerr_terms::Int = 1
    ) -> Float64

Unified interface for estimating an axis-separable midpoint-residual truncation-error *model*
in arbitrary dimensions.

# Function description
Dispatches to the corresponding dimension-specific estimator:
- `dim == 1` ``\\rightarrow`` [`error_estimate_1d`](@ref)
- `dim == 2` ``\\rightarrow`` [`error_estimate_2d`](@ref)
- `dim == 3` ``\\rightarrow`` [`error_estimate_3d`](@ref)
- `dim == 4` ``\\rightarrow`` [`error_estimate_4d`](@ref)
- `dim >= 5` ``\\rightarrow`` [`error_estimate_nd`](@ref)

All estimators use the exact midpoint residual expansion derived from rational weight assembly
for NS-style composite rules. When `nerr_terms > 1`, the model includes LO plus additional
nonzero midpoint residual terms (LO+NLO+...).

# Arguments
- `f`:
    Integrand function (expects `dim` positional arguments).
- `a`, `b`:
    Bounds for each dimension (interpreted as scalar bounds for a hypercube ``[a,b]^\\texttt{dim}``).
- `N`:
    Number of subdivisions per axis (subject to rule constraints in the 1D case).
- `dim`:
    Number of dimensions (`Int`).
- `rule`:
    Integration rule symbol (must be `:ns_pK` style for the residual-based model).
- `boundary`:
    Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Keyword arguments
- `nerr_terms`:
    Number of nonzero midpoint residual terms to include (`1` = LO only, `2` = LO+NLO, ...).

# Returns
- `Float64`:
    A multidimensional truncation-error model value (axis-separable; mixed-derivative terms are omitted).
"""
function error_estimate(
    f, 
    a, 
    b, 
    N, 
    dim, 
    rule,
    boundary;
    nerr_terms::Int = 1
)
    if dim == 1
        return error_estimate_1d(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 2
        return error_estimate_2d(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 3
        return error_estimate_3d(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 4
        return error_estimate_4d(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    else
        return error_estimate_nd(f, a, b, N, rule, boundary; dim=dim, nerr_terms=nerr_terms)
    end
end

"""
    error_estimate_threads(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        nerr_terms::Int = 1
    ) -> Float64

Threaded dispatcher for the axis-separable midpoint-residual truncation-error *model* in arbitrary dimensions.

# Function description
Dispatches to the corresponding **threaded** dimension-specific estimator:
- `dim == 1` ``\\rightarrow`` [`error_estimate_1d_threads`](@ref)
- `dim == 2` ``\\rightarrow`` [`error_estimate_2d_threads`](@ref)
- `dim == 3` ``\\rightarrow`` [`error_estimate_3d_threads`](@ref)
- `dim == 4` ``\\rightarrow`` [`error_estimate_4d_threads`](@ref)
- `dim >= 5` ``\\rightarrow`` [`error_estimate_nd_threads`](@ref)

All non-threading details (mathematical definition, coefficient construction, residual-term
interpretation, and overall intent) are identical to [`error_estimate`](@ref).
See that function for the full formalism and background. Threading strategy details are
documented in each dimension-specific threaded estimator.

# Arguments
Same as [`error_estimate`](@ref).

# Keyword arguments
Same as [`error_estimate`](@ref).

# Returns
Same as [`error_estimate`](@ref).
"""
function error_estimate_threads(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    nerr_terms::Int = 1
)
    if dim == 1
        return error_estimate_1d_threads(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 2
        return error_estimate_2d_threads(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 3
        return error_estimate_3d_threads(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    elseif dim == 4
        return error_estimate_4d_threads(f, a, b, N, rule, boundary, nerr_terms=nerr_terms)
    else
        return error_estimate_nd_threads(f, a, b, N, rule, boundary; dim=dim, nerr_terms=nerr_terms)
    end
end

end  # module ErrorDispatch