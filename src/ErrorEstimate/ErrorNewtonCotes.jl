# ============================================================================
# src/ErrorEstimate/ErrorNewtonCotes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================


module ErrorNewtonCotes

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.QuadratureDispatch

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
Newton-Cotes rule assembled on the integer ``u``-grid ``j = 0 , \\ldots , N_{\\text{sub}}``:

- Center (dimensionless): ``\\displaystyle{c = \\frac{N_{\\text{sub}}}{2}}``
- Residual moment for each ``k``:
```math
\\texttt{\\texttt{diff}}_k = \\int\\limits_{0}^{N_{\\text{sub}}} du \\; \\left( u - c \\right)^k - \\sum_{j=0}^{N_{\\text{sub}}} \\beta_j \\, \\left( j - c \\right)^k 
```

- Taylor coefficient (exact rational):
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}
```

The function returns the *first* ``k`` (starting from ``k=0``) for which ``\\texttt{diff}_k \\neq 0``,
together with its exact rational coefficient ``\\texttt{coeff}_k``.

This is intended to identify the leading midpoint residual order used by the
midpoint-based convergence/error heuristics.

# Arguments
- `β`: Exact rational composite coefficient vector ``\\beta`` (length ``N_{\\text{sub}} + 1``), typically produced by
  [`NewtonCotes._assemble_composite_beta_rational`](@ref). The entry `β[j+1]` corresponds to node ``j``.
- `Nsub`: Number of subintervals defining the `u`-grid ``0, \\ldots, N_{\\text{sub}}``.
- `kmax`: Maximum derivative order ``k`` to scan (inclusive).

# Returns
- `k::Int`: The first order with nonzero residual moment.
- `coeff::NewtonCotes.RBig`: The exact rational Taylor coefficient ``\\displaystyle{\\frac{\\texttt{diff}_k}{k!}}``.

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
This helper is a convenience wrapper around
[`NewtonCotes._assemble_composite_beta_rational`](@ref) and
[`_leading_midpoint_residual_term_from_beta`](@ref).

Workflow:
1) Validate `boundary` via [`QuadratureDispatch._decode_boundary`](@ref).
2) Require `rule` to be of Newton-Cotes form `:newton_pK` (midpoint residual model is defined for these).
3) Parse ``p`` from `rule` and assemble the exact rational ``\\beta``.
4) Scan for the first nonzero midpoint residual term `(k, coeff)`.

# Arguments
- `rule`: Must be an `:newton_pK`-style rule symbol (e.g. `:newton_p3`, `:newton_p5`).
- `boundary`: Boundary pattern (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
- `Nsub`: Number of subintervals (must satisfy the composite tiling constraint for that boundary).
- `kmax`: Maximum ``k`` to scan.

# Returns
- `(k, coeff)`: The leading nonzero residual order and its exact coefficient.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid, if `rule` is not `:newton_pK`,
  if `Nsub` is invalid for the boundary tiling, or if no term is found up to `kmax`.
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

Collect the first `nterms` nonzero midpoint residual orders `k` (ascending) and report the expansion center.

# Function description
This helper searches the midpoint-centered residual moments for a composite Newton-Cotes rule,
and returns the first `nterms` derivative orders `k` for which the residual moment is nonzero.

It uses:
- Center: ``\\displaystyle{c = \\frac{N_{\\text{sub}}}{2} }`` (returned as `:mid`)
- Residual test:
```math
\\texttt{\\texttt{diff}}_k = \\int\\limits_{0}^{N_{\\text{sub}}} du \\; \\left( u - c \\right)^k - \\sum_{j=0}^{N_{\\text{sub}}} \\beta_j \\, \\left( j - c \\right)^k 
```

If ``\\texttt{diff}_k \\neq 0``, then ``k`` is recorded.

This is useful for constructing multi-term convergence models where multiple
nonzero residual orders are needed (*e.g.* fitting several powers).

# Arguments
- `rule`: `:newton_pK` rule symbol (required by the current implementation).
- `boundary`: Boundary pattern (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
- `Nsub`: Number of subintervals for the composite rule.
- `nterms`: Number of nonzero `k` values to collect (must satisfy `nterms ≥ 1`).
- `kmax`: Maximum `k` to scan (inclusive).

# Returns
- `ks::Vector{Int}`: First `nterms` residual derivative orders, sorted by scan order (ascending).
- `center::Symbol`: Expansion center indicator. Currently always `:mid`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if it cannot collect `nterms` values up to `kmax`.
- Propagates errors from exact ``\\beta`` assembly if `(rule, boundary, Nsub)` is invalid.
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
This routine generalizes [`_leading_midpoint_residual_term_from_beta`](@ref) by collecting
multiple nonzero residual terms.

For each ``k = 0 , \\ldots , k_{\\max}``, it computes:
- Center: ``\\displaystyle{c = \\frac{N_{\\text{sub}}}{2} }``
- Exact moment:
```math
\\texttt{exact}_k = \\int\\limits_0^{N_{\\text{sub}}} du \\; \\left( u - c \\right)^k
```

- Quadrature moment:
```math
\\texttt{approx}_k = \\sum_0^{N_{\\text{sub}}} \\beta_j \\, \\left( j - c \\right)^k
```

- Residual and Taylor coefficient:
```math
\\texttt{diff}_k = \\texttt{exact}_k - \\texttt{approx}_k
```
```math
\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}
```
Whenever ``\\texttt{diff}_k \\neq 0``, the pair `(k, coeff(k))` is appended.
The function stops once `nterms` pairs have been collected.

All outputs remain in exact rational arithmetic ([`NewtonCotes.RBig`](@ref)).

# Arguments
- `β`: Exact rational coefficient vector of length ``N_\\text{sub} + 1`` (index `β[j+1]` corresponds to node `j`).
- `Nsub`: Number of subintervals defining the ``u``-grid ``0 , \\ldots , N_{\\text{sub}}``.
- `nterms`: Number of nonzero terms to collect (must satisfy `nterms ≥ 1`).
- `kmax`: Maximum `k` to scan (inclusive, must satisfy `kmax ≥ 0`).

# Returns
- `ks::Vector{Int}`: Collected residual orders `k` (ascending by scan).
- `coeffs::Vector{NewtonCotes.RBig}`: Exact Taylor coefficients `diff(k)/k!`, aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1` or `kmax < 0`.
- Throws if fewer than `nterms` nonzero terms exist up to `kmax`.
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
This helper is the public-facing (within `ErrorEstimate`) convenience wrapper
for midpoint residual extraction.

Workflow:
1) Validate `boundary` via [`QuadratureDispatch._decode_boundary`](@ref) (catches typos early).
2) Require `rule` to be an Newton-Cotes rule (`:newton_pK`) because the residual/``\\beta`` construction
   is defined in terms of the exact-rational Newton-Cotes assembly.
3) Parse `p` from `rule`.
4) Assemble exact rational composite coefficients `βR` using
   [`NewtonCotes._assemble_composite_beta_rational`](@ref)`(p, boundary, Nsub)`.
5) Extract the first `nterms` nonzero midpoint residual pairs `(k, coeff)` via
   [`_leading_midpoint_residual_terms_from_beta`](@ref).

# Arguments
- `rule`: Must be an `:newton_pK` rule symbol.
- `boundary`: Boundary pattern (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
- `Nsub`: Number of subintervals for the composite rule.
- `nterms`: Number of nonzero residual terms to collect (must satisfy `nterms ≥ 1`).
- `kmax`: Maximum derivative order scanned (inclusive, must satisfy `kmax ≥ 0`).

# Returns
- `ks::Vector{Int}`: First `nterms` nonzero residual orders.
- `coeffs::Vector{NewtonCotes.RBig}`: Exact rational coefficients ``\\displaystyle{\\frac{\\texttt{diff}_k}{k!}}`` `diff(k)/k!` aligned with `ks`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid, if `rule` is not `:newton_pK`,
  if `Nsub` violates composite constraints, or if insufficient nonzero terms exist up to `kmax`.
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

end  # module ErrorNewtonCotes