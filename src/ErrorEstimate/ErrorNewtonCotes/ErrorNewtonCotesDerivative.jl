# ============================================================================
# src/ErrorEstimate/ErrorNewtonCotesDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorNewtonCotesDerivative

import ..JobLoggerTools
import ..NewtonCotes
import ..QuadratureDispatch

# ----------------------------
# helper: collect first nonzero midpoint residual term
# E = Σ_k coeff(k) * h^(k+1) * f^(k)(x_mid)
# coeff(k) = ( ∫_0^N (u-c)^k du - Σ β[j] (j-c)^k ) / k!
# ----------------------------

"""
    _leading_midpoint_residual_term_from_beta(
        β::Vector{NewtonCotes.RBig},
        Nsub::Int;
        kmax::Int = 64
    ) -> Tuple{Int, NewtonCotes.RBig}

Find the first nonzero midpoint residual term `(k, coeff)` from exact composite weights ``\\beta``.

# Function description
This helper scans the midpoint-centered residual expansion induced by a composite
Newton-Cotes rule assembled on the integer ``u``-grid ``j = 0, \\ldots, N_{\\texttt{sub}}``.

Using the midpoint
```math
c = \\frac{N_{\\texttt{sub}}}{2},
```
it computes, for each `k`,
```math
\\texttt{diff}_k
=
\\int\\limits_0^{N_{\\texttt{sub}}} (u-c)^k \\, du
-
\\sum_{j=0}^{N_{\\texttt{sub}}} \\beta_j (j-c)^k,
```
and the exact rational Taylor coefficient
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}.
```

The first `k` with `diff_k != 0` is returned together with ``\\texttt{coeff}_k``.

# Arguments
- `β`: Exact rational composite coefficient vector ``\\beta`` of length ``N_{\\texttt{sub}}+1``.
- `Nsub`: Number of subintervals on the dimensionless composite grid.
- `kmax`: Maximum order to scan.

# Returns
- `k::Int`: First order with nonzero midpoint residual.
- `coeff::NewtonCotes.RBig`: Exact rational coefficient ``\\dfrac{\\texttt{diff}_k}{k!}``.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `kmax < 0`.
- Throws if no nonzero residual is found up to `kmax`.
"""
function _leading_midpoint_residual_term_from_beta(
    β::Vector{NewtonCotes.RBig},
    Nsub::Int;
    kmax::Int = 64
)::Tuple{Int, NewtonCotes.RBig}

    kmax >= 0 || JobLoggerTools.error_benji("kmax must be ≥ 0")

    c = NewtonCotes.RBig(BigInt(Nsub), 2)  # Nsub/2
    Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)

    for k in 0:kmax
        # exact = ∫_0^{Nsub} (u-c)^k du
        exact = ((Nrb - c)^(k+1) - (NewtonCotes.RBig(0) - c)^(k+1)) / NewtonCotes.RBig(BigInt(k+1), 1)

        approx = NewtonCotes.RBig(0)
        @inbounds for j in 0:Nsub
            wj = β[j+1]
            wj == 0 && continue
            approx += wj * (NewtonCotes.RBig(BigInt(j), 1) - c)^k
        end

        diff = exact - approx
        if diff != 0
            coeff = diff / NewtonCotes.RBig(factorial(big(k)), 1)
            return k, coeff
        end
    end

    JobLoggerTools.error_benji("Could not find a nonzero midpoint residual term up to kmax=$kmax (Nsub=$Nsub).")
end

"""
    _leading_midpoint_residual_term(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        kmax::Int = 64
    ) -> Tuple{Int, NewtonCotes.RBig}

Build exact composite weights for `(rule, boundary, Nsub)` and extract the leading midpoint residual term.

# Function description
This is a convenience wrapper around
[`NewtonCotes._assemble_composite_beta_rational`](@ref) and
[`_leading_midpoint_residual_term_from_beta`](@ref).

It validates the boundary, requires a `:newton_p2`, `:newton_p3`... rule, assembles the exact
rational coefficient vector ``\\beta``, and returns the first nonzero midpoint
residual term `(k, coeff)`.

# Arguments
- `rule`: Newton-Cotes rule symbol of the form `:newton_p2`, `:newton_p3`, etc.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of composite subintervals.
- `kmax`: Maximum order to scan.

# Returns
- `Tuple{Int, NewtonCotes.RBig}`: Leading residual order and exact coefficient.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid.
- Throws if `rule` is not of the form `:newton_p2`, `:newton_p3`, etc.
- Propagates composite-assembly errors and failure-to-find-term errors.
"""
function _leading_midpoint_residual_term(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    kmax::Int = 64
)::Tuple{Int, NewtonCotes.RBig}

    # boundary validation (also catches typos early)
    QuadratureDispatch._decode_boundary(boundary)

    NewtonCotes._is_newton_cotes_rule(rule) || JobLoggerTools.error_benji("midpoint residual model currently expects :newton_pK rules (got rule=$rule)")

    p = NewtonCotes._parse_newton_p(rule)

    # exact β (rational) from your assembly
    βR = NewtonCotes._assemble_composite_beta_rational(p, boundary, Nsub)

    return _leading_midpoint_residual_term_from_beta(βR, Nsub; kmax=kmax)
end

"""
    _leading_residual_ks_with_center(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int,
        kmax::Int = 128
    ) -> Tuple{Vector{Int}, Symbol}

Collect the first `nterms` nonzero midpoint residual orders and report the expansion center.

# Function description
This helper scans midpoint-centered residual moments for a composite Newton-Cotes
rule and records the first `nterms` derivative orders ``k`` for which the residual
moment is nonzero.

The returned center tag is currently always `:mid`.

# Arguments
- `rule`: Newton-Cotes rule symbol `:newton_p2`, `:newton_p3`, etc.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of composite subintervals.
- `nterms`: Number of nonzero orders to collect.
- `kmax`: Maximum order to scan.

# Returns
- `ks::Vector{Int}`: First `nterms` nonzero residual orders in ascending scan order.
- `center::Symbol`: Expansion center tag, currently `:mid`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if insufficient terms are found up to `kmax`.
- Propagates errors from exact composite assembly.
"""
function _leading_residual_ks_with_center(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int,
    kmax::Int=128
)::Tuple{Vector{Int}, Symbol}

    ks = Int[]
    k = 0
    center = :mid

    while length(ks) < nterms
        β = NewtonCotes._assemble_composite_beta_rational(NewtonCotes._parse_newton_p(rule), boundary, Nsub)
        c = NewtonCotes.RBig(BigInt(Nsub), 2)
        # scan k upward and collect first n nonzero midpoint residuals
        center = :mid
        while length(ks) < nterms && k <= kmax
            # exact ∫ (u-c)^k
            exact = ((NewtonCotes.RBig(BigInt(Nsub),1) - c)^(k+1) - (NewtonCotes.RBig(0) - c)^(k+1)) /
                    NewtonCotes.RBig(BigInt(k+1),1)
            approx = NewtonCotes.RBig(0)
            for j in 0:Nsub
                wj = β[j+1]; wj == 0 && continue
                approx += wj * (NewtonCotes.RBig(BigInt(j),1) - c)^k
            end
            if exact - approx != 0
                push!(ks, k)
            end
            k += 1
        end
        break
    end

    (length(ks) == nterms) || JobLoggerTools.error_benji("Could not collect nterms=$nterms residual ks up to kmax=$kmax")
    return ks, center
end

"""
    _leading_midpoint_residual_terms_from_beta(
        β::Vector{NewtonCotes.RBig},
        Nsub::Int;
        nterms::Int = 2,
        kmax::Int = 128
    ) -> Tuple{Vector{Int}, Vector{NewtonCotes.RBig}}

Collect the first `nterms` nonzero midpoint residual terms `(k, coeff)` from exact weights ``\\beta``.

# Function description
This generalizes [`_leading_midpoint_residual_term_from_beta`](@ref) by collecting
multiple nonzero midpoint residual terms.

For each scanned order ``k``, it computes the exact midpoint-centered moment,
the quadrature moment induced by ``\\beta``, and the Taylor coefficient
``\\dfrac{\\texttt{diff}_k}{k!}``. 
Every nonzero residual is appended until `nterms` terms have
been collected.

# Arguments
- `β`: Exact rational coefficient vector of length ``N_{\\texttt{sub}}+1``.
- `Nsub`: Number of subintervals on the dimensionless composite grid.
- `nterms`: Number of nonzero terms to collect.
- `kmax`: Maximum order to scan.

# Returns
- `ks::Vector{Int}`: Collected residual orders in ascending scan order.
- `coeffs::Vector{NewtonCotes.RBig}`: Exact rational coefficients aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1` or `kmax < 0`.
- Throws if fewer than `nterms` nonzero terms are found up to `kmax`.
"""
function _leading_midpoint_residual_terms_from_beta(
    β::Vector{NewtonCotes.RBig},
    Nsub::Int;
    nterms::Int = 2,
    kmax::Int = 128
)::Tuple{Vector{Int}, Vector{NewtonCotes.RBig}}

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0")

    c   = NewtonCotes.RBig(BigInt(Nsub), 2)  # c = Nsub/2
    Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)  # N as rational

    ks     = Int[]
    coeffs = NewtonCotes.RBig[]

    for k in 0:kmax
        # Exact moment: ∫_0^N (u-c)^k du
        exact = ((Nrb - c)^(k+1) - (NewtonCotes.RBig(0) - c)^(k+1)) /
                NewtonCotes.RBig(BigInt(k+1), 1)

        # Quadrature moment: Σ β[j] * (j-c)^k
        approx = NewtonCotes.RBig(0)
        @inbounds for j in 0:Nsub
            wj = β[j+1]
            wj == 0 && continue
            approx += wj * (NewtonCotes.RBig(BigInt(j), 1) - c)^k
        end

        diff = exact - approx
        if diff != 0
            # Convert moment residual to Taylor coefficient: diff/k!
            coeff = diff / NewtonCotes.RBig(factorial(big(k)), 1)
            push!(ks, k)
            push!(coeffs, coeff)
            length(ks) == nterms && return ks, coeffs
        end
    end

    JobLoggerTools.error_benji(
        "Could not collect nterms=$nterms midpoint residual terms up to kmax=$kmax (Nsub=$Nsub)."
    )
end

"""
    _leading_midpoint_residual_terms(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 2,
        kmax::Int = 128
    ) -> Tuple{Vector{Int}, Vector{NewtonCotes.RBig}}

Build exact composite weights for `(rule, boundary, Nsub)` and collect the first `nterms` midpoint residual terms.

# Function description
This helper validates the boundary, requires a `:newton_p2`, `:newton_p3`... rule, assembles the
exact rational composite coefficients, and extracts the first `nterms` nonzero
midpoint residual pairs `(k, coeff)`.

# Arguments
- `rule`: Newton-Cotes rule symbol `:newton_p2`, `:newton_p3`, etc.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of composite subintervals.
- `nterms`: Number of nonzero residual terms to collect.
- `kmax`: Maximum order to scan.

# Returns
- `ks::Vector{Int}`: First `nterms` nonzero midpoint residual orders.
- `coeffs::Vector{NewtonCotes.RBig}`: Exact rational coefficients aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid.
- Throws if `rule` is not of the form `:newton_pK`.
- Propagates composite-assembly errors and insufficient-term errors.
"""
function _leading_midpoint_residual_terms(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 2,
    kmax::Int = 128
)::Tuple{Vector{Int}, Vector{NewtonCotes.RBig}}

    # Validate boundary symbol (also catches typos early)
    QuadratureDispatch._decode_boundary(boundary)

    # This residual construction currently assumes ns rules
    NewtonCotes._is_newton_cotes_rule(rule) || JobLoggerTools.error_benji(
        "midpoint residual model currently expects :newton_pK rules (got rule=$rule)"
    )

    p  = NewtonCotes._parse_newton_p(rule)

    # Exact rational global weights β[0..Nsub] for the chosen boundary pattern
    βR = NewtonCotes._assemble_composite_beta_rational(p, boundary, Nsub)

    return _leading_midpoint_residual_terms_from_beta(βR, Nsub; nterms=nterms, kmax=kmax)
end

end  # module ErrorNewtonCotesDerivative