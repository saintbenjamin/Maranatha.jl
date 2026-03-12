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
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    )

Estimate a ``1``-dimensional midpoint-residual truncation-error model.

# Function description
This routine builds the axis-separable asymptotic error model for the
``1``-dimensional case using the leading nonzero midpoint residual terms of the
selected composite rule.

For each collected residual order ``k_i``, it evaluates the ``k_i``-th derivative of
``f`` at the physical midpoint and forms the contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, f^{(k_i)}(\\bar{x}) \\, .
```

# Arguments
- `f`: Scalar callable integrand ``f(x)``.
- `a::Real`: Lower bound.
- `b::Real`: Upper bound.
- `N::Int`: Number of subintervals.
- `rule::Symbol`: Quadrature rule symbol.
- `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments
- `err_method::Symbol`: Derivative backend selector.
- `nerr_terms::Int`: Number of nonzero residual terms to include.
- `kmax::Int`: Maximum residual order scanned.

# Returns
- `NamedTuple` with fields:
  - `ks`
  - `coeffs`
  - `derivatives`
  - `terms`
  - `total`
  - `center`
  - `h`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or `kmax < 0`.
- Propagates residual-extraction and derivative-evaluation errors.

# Notes
- This is an asymptotic error model, not a rigorous bound.
- A detected `k == 0` term is suppressed in the returned contribution.
"""
function error_estimate_1d(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N
    x̄ = (aa + bb) / 2

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    @inbounds for i in eachindex(ks)
        k = ks[i]

        if k == 0
            derivatives[i] = 0.0
            terms[i] = 0.0
            continue
        end

        coeff = coeffs[i]

        dx = nth_derivative(
            f,
            x̄, k;
            h=h, rule=rule, N=N, dim=1,
            side=:mid, axis=:x, stage=:midpoint,
            err_method=err_method
        )

        derivatives[i] = dx
        terms[i] = coeff * h^(k + 1) * dx
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = x̄,
        h           = h
    )
end

"""
    error_estimate_1d_threads(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128
    )

Threaded variant of [`error_estimate_1d`](@ref).

# Function description
This routine keeps the same mathematical model as [`error_estimate_1d`](@ref) but
parallelizes the loop over collected residual terms using Julia threading.

Each term is evaluated independently, and the final result is assembled into the
same return structure as the non-threaded version.

# Arguments
- Same as [`error_estimate_1d`](@ref).

# Keyword arguments
- Same as [`error_estimate_1d`](@ref).

# Returns
- Same `NamedTuple` structure as [`error_estimate_1d`](@ref).

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or `kmax < 0`.
- Propagates residual-extraction and derivative-evaluation errors.

# Notes
- Threading overhead may dominate when only a small number of residual terms is used.
"""
function error_estimate_1d_threads(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N
    x̄ = (aa + bb) / 2

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    Threads.@threads for i in eachindex(ks)
        k = ks[i]

        if k == 0
            derivatives[i] = 0.0
            terms[i] = 0.0
            continue
        end

        coeff = coeffs[i]

        dx = nth_derivative(
            f, x̄, k;
            h=h, rule=rule, N=N, dim=1,
            side=:mid, axis=:x, stage=:midpoint,
            err_method=err_method
        )

        derivatives[i] = dx
        terms[i] = coeff * h^(k + 1) * dx
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = x̄,
        h           = h
    )
end