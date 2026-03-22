# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_leading_residual_ks_with_center_any.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _leading_residual_ks_with_center_any(
        rule::Symbol,
        boundary,
        Nsub::Int;
        fit_func_terms::Int,
        kmax::Int = 256,
        tol_abs = nothing,
        tol_rel = nothing,
        real_type = Float64,
    ) -> Tuple{Vector{Int}, Symbol}

Extract only the residual indices `k` of the first `fit_func_terms` nonzero midpoint-shifted
residual moments, together with the centering convention.

# Function description
This is a lighter-weight companion to [`_leading_residual_terms_any`](@ref) that
returns only the detected residual orders and the center tag.

Supported backends:

- Newton-Cotes rules: exact rational residual detection.
- Gauss-family rules: tolerance-based `Float64` residual detection.
- B-spline rules: tolerance-based `Float64` residual detection.

The center is currently always `:mid`.

# Arguments
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern specification. If a tuple/vector is supplied,
  only its first entry is used.
- `Nsub`: Number of unit blocks in the dimensionless tiling domain.

# Keyword arguments
- `fit_func_terms`: Number of residual indices to collect.
- `kmax`: Maximum moment order to scan.
- `tol_abs`: Absolute tolerance for floating-point residual detection. If `nothing`,
  a type-scaled default is used.
- `tol_rel`: Relative tolerance for floating-point residual detection. If `nothing`,
  a type-scaled default is used.
- `real_type = Float64`:
  Scalar type used internally for floating-point residual tests.

# Returns
- `ks::Vector{Int}`: Residual indices where a nonzero moment was detected.
- `center::Symbol`: Centering convention tag, currently `:mid`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `fit_func_terms < 1` or `kmax < 0`.
- Throws if `boundary` is invalid or `rule` is unsupported.
- Propagates backend residual-detection errors.
"""
function _leading_residual_ks_with_center_any(
    rule::Symbol,
    boundary,
    Nsub::Int;
    fit_func_terms::Int,
    kmax::Int = 256,
    tol_abs = nothing,
    tol_rel = nothing,
    real_type = Float64,
)::Tuple{Vector{Int}, Symbol}

    T = real_type
    tol_abs_T = isnothing(tol_abs) ? T(5e4) * eps(T) : convert(T, tol_abs)
    tol_rel_T = isnothing(tol_rel) ? T(5e4) * eps(T) : convert(T, tol_rel)

    (fit_func_terms >= 1) || JobLoggerTools.error_benji(
        "fit_func_terms must be ≥ 1 (got $fit_func_terms)"
    )
    (kmax >= 0) || JobLoggerTools.error_benji(
        "kmax must be ≥ 0 (got $kmax)"
    )

    bd = if boundary isa Symbol
        QuadratureBoundarySpec._decode_boundary(boundary)
        boundary
    elseif boundary isa Tuple || boundary isa AbstractVector
        isempty(boundary) && throw(ArgumentError("boundary must not be empty"))
        boundary[1] isa Symbol || throw(ArgumentError("boundary[1] must be a Symbol"))
        QuadratureBoundarySpec._decode_boundary(boundary[1])
        boundary[1]
    else
        throw(ArgumentError("unsupported boundary specification"))
    end
    QuadratureBoundarySpec._decode_boundary(bd)
    center = :mid

    if NewtonCotes._is_newton_cotes_rule(rule)
        ks = Int[]
        c  = NewtonCotes.RBig(BigInt(Nsub), 2)
        Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)

        β = NewtonCotes._assemble_composite_beta_rational(
            NewtonCotes._parse_newton_p(rule), bd, Nsub
        )

        for k in 0:kmax
            exact = ((Nrb - c)^(k + 1) - (NewtonCotes.RBig(0) - c)^(k + 1)) /
                    NewtonCotes.RBig(BigInt(k + 1), 1)

            approx = NewtonCotes.RBig(0)
            @inbounds for j in 0:Nsub
                wj = β[j + 1]
                wj == 0 && continue
                approx += wj * (NewtonCotes.RBig(BigInt(j), 1) - c)^k
            end

            if exact - approx != 0
                push!(ks, k)
                length(ks) == fit_func_terms && return ks, center
            end
        end

        JobLoggerTools.error_benji(
            "Could not collect fit_func_terms=$fit_func_terms Newton-Cotes residual ks up to kmax=$kmax"
        )
    end

    if Gauss._is_gauss_rule(rule)
        npts = Gauss._parse_gauss_p(rule)

        if bd === :LU_INEX || bd === :LU_EXIN
            ks = Int[]
            k0 = 2 * npts - 1
            for j in 0:(fit_func_terms - 1)
                push!(ks, k0 + 2 * j)
            end
            return ks, center
        end

        U, W = Gauss._composite_gauss_u_grid(Nsub, npts, bd)
        UT = T.(U)
        WT = T.(W)
        c = T(Nsub) / T(2)

        ks = Int[]
        for k in 1:kmax
            exact = ((T(Nsub) - c)^(k + 1) - (zero(T) - c)^(k + 1)) / T(k + 1)

            approx = zero(T)
            @inbounds for i in eachindex(UT)
                approx += WT[i] * (UT[i] - c)^k
            end

            diff = exact - approx

            if abs(diff) > (tol_abs_T + tol_rel_T * abs(exact))
                push!(ks, k)
                length(ks) == fit_func_terms && return ks, center
            end
        end

        if !isempty(ks) && ks[1] == 0
            JobLoggerTools.error_benji(
                "Gauss residual ks starts with 0 (unstable moment-test). ks=$ks rule=$rule boundary=$bd Nsub=$Nsub"
            )
        end

        JobLoggerTools.error_benji(
            "Could not collect fit_func_terms=$fit_func_terms Gauss residual ks up to kmax=$kmax"
        )
    end

    if BSpline._is_bspline_rule(rule)
        ks, center = ErrorBSplineDerivative._leading_residual_ks_with_center_bspline_float(
            rule,
            bd,
            Nsub;
            fit_func_terms = fit_func_terms,
            kmax = kmax,
            λ = 0.0,
            tol_abs = Float64(tol_abs_T),
            tol_rel = Float64(tol_rel_T),
        )
        return ks, center
    end

    JobLoggerTools.error_benji(
        "Unsupported rule=$rule for residual ks extraction."
    )
end
