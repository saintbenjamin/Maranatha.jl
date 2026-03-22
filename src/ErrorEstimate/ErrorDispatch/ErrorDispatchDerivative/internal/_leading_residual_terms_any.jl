# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_leading_residual_terms_any.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _leading_residual_terms_any(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 1,
        kmax::Int = 128,
        real_type = Float64,
    ) -> Tuple{Vector{Int}, Vector, Symbol}

Collect the first `nterms` nonzero midpoint-shifted residual coefficients
for a supported quadrature backend.

# Function description
This helper normalizes the currently supported residual backends into a common
return type:

- Newton-Cotes rules use the exact-rational residual backend and convert the
  resulting coefficients to the requested `real_type`.
- Gauss-family rules use the `Float64` midpoint-residual backend directly.
- B-spline rules use the `Float64` midpoint-residual backend directly.

The returned `center` tag is currently always `:mid`.

# Arguments
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of unit blocks in the dimensionless tiling domain.

# Keyword arguments
- `nterms`: Number of leading nonzero residual terms to return.
- `kmax`: Maximum moment order to scan.
- `real_type = Float64`:
  Scalar type used for coefficient conversion in the unified return value.

# Returns
- `ks::Vector{Int}`: Residual indices where a nonzero moment was detected.
- `coeffs`: Factorial-scaled residual coefficients in the active scalar type.
- `center::Symbol`: Centering convention tag, currently `:mid`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid.
- Throws if `rule` is unsupported.
- Propagates backend errors if the requested number of terms cannot be collected.
"""
function _leading_residual_terms_any(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int = 1,
    kmax::Int = 128,
    real_type = Float64,
)::Tuple{Vector{Int}, Vector, Symbol}

    T = real_type
    QuadratureBoundarySpec._decode_boundary(boundary)

    if NewtonCotes._is_newton_cotes_rule(rule)
        if nterms == 1
            k, coeffR = ErrorNewtonCotesDerivative._leading_midpoint_residual_term(
                rule,
                boundary,
                Nsub;
                kmax = min(kmax, 64),
            )
            return [k], T[convert(T, coeffR)], :mid
        else
            ks, coeffsR = ErrorNewtonCotesDerivative._leading_midpoint_residual_terms(
                rule,
                boundary,
                Nsub;
                nterms = nterms,
                kmax = kmax,
            )
            return ks, T.(coeffsR), :mid
        end
    end

    if Gauss._is_gauss_rule(rule)
        ks, coeffs = ErrorGaussDerivative._leading_midpoint_residual_terms_gauss_float(
            rule,
            boundary,
            Nsub;
            nterms = nterms,
            kmax = kmax,
        )
        return ks, T.(coeffs), :mid
    end

    if BSpline._is_bspline_rule(rule)
        ks, coeffs = ErrorBSplineDerivative._leading_midpoint_residual_terms_bspline_float(
            rule,
            boundary,
            Nsub;
            nterms = nterms,
            kmax = kmax,
            λ = 0.0,
        )
        return ks, T.(coeffs), :mid
    end

    JobLoggerTools.error_benji(
        "Unsupported rule for residual model: rule=$rule"
    )
end
