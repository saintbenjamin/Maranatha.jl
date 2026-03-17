# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet_1d(
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

Estimate a ``1``-dimensional midpoint-residual truncation-error model using
derivative-jet reuse.

# Function description
This routine builds the same ``1``-dimensional asymptotic midpoint-residual
error model as [`error_estimate_derivative_direct_1d`](@ref), but instead of requesting each
derivative order independently, it evaluates the required derivatives through a
shared derivative jet.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function calls [`AutoDerivativeJet._derivative_values_for_ks`](@ref) to obtain
all requested midpoint derivatives from a single jet-based evaluation path.

For each collected residual order ``k_i``, it forms the contribution
```math
E \\approx \\sum_{i=1}^{n_{\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, f^{(k_i)}(\bar{x}) \\, .
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
- Propagates residual-model extraction and jet-based derivative-evaluation
  errors.

# Notes
- This is an asymptotic error model, not a rigorous bound.
- A detected `k == 0` term is suppressed in the returned contribution.
- If `ks` is empty, the function returns a zero-total result with empty arrays.
- This variant is especially useful when multiple derivative orders are needed,
  since one shared jet can serve all requested residual terms.
"""
function error_estimate_derivative_jet_1d(
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
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N
    x̄ = (aa + bb) / 2

    ks, coeffs, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    n = length(ks)

    derivatives = Vector{Float64}(undef, n)
    terms       = Vector{Float64}(undef, n)

    isempty(ks) && return (;
        ks          = Int[],
        coeffs      = Float64[],
        derivatives = Float64[],
        terms       = Float64[],
        total       = 0.0,
        center      = x̄,
        h           = h
    )

    vals = AutoDerivativeJet._derivative_values_for_ks(
        f,
        x̄,
        ks;
        h = h,
        rule = rule,
        N = N,
        dim = 1,
        err_method = err_method,
        side = :mid,
        axis = :x,
        stage = :midpoint,
    )

    @inbounds for i in eachindex(ks)
        k = ks[i]

        if k == 0
            derivatives[i] = 0.0
            terms[i] = 0.0
            continue
        end

        derivatives[i] = vals[i]
        terms[i] = coeffs[i] * h^(k + 1) * derivatives[i]
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