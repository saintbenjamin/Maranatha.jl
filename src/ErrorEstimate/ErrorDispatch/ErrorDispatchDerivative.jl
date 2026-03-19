# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorDispatchDerivative

import ..JobLoggerTools
import ..NewtonCotes
import ..Gauss
import ..BSpline
import ..QuadratureUtils
import ..AutoDerivativeDirect
import ..AutoDerivativeJet
import ..ErrorNewtonCotesDerivative
import ..ErrorGaussDerivative
import ..ErrorBSplineDerivative
import .._RES_MODEL_CACHE
import .._NTH_DERIV_CACHE
import .._DERIV_JET_CACHE

"""
    clear_error_estimate_derivative_caches!() -> Nothing

Clear all global caches used by the error-estimation layer.

# Function description
This helper empties the residual-model cache, derivative-value cache, and
derivative-jet cache, then prints the resulting cache sizes through
[`JobLoggerTools.println_benji`](@ref).

This is useful when:

- benchmarking cache behavior,
- forcing a clean recomputation,
- debugging stale cache contents,
- resetting state between large runs.

# Returns
- `nothing`

# Side effects
- Mutates the following global caches:
  - [`_RES_MODEL_CACHE`](@ref)
  - [`_NTH_DERIV_CACHE`](@ref)
  - [`_DERIV_JET_CACHE`](@ref)
- Emits diagnostic log lines showing the new cache sizes.

# Notes
- The printed sizes should normally all be zero immediately after this call.
"""
function clear_error_estimate_derivative_caches!()
    empty!(_RES_MODEL_CACHE)
    empty!(_NTH_DERIV_CACHE)
    empty!(_DERIV_JET_CACHE)
    return nothing
end

"""
    _get_residual_model_fixed(
        rule::Symbol,
        boundary::Symbol,
        Nref::Int;
        nterms::Int,
        kmax::Int,
        real_type = Float64,
    ) -> Tuple{Vector{Int}, Vector, Symbol}

Return a cached residual model for a fixed quadrature configuration.

# Function description
This helper retrieves the leading residual-term model associated with a given
quadrature rule and boundary pattern. If a matching model is already present in
[`_RES_MODEL_CACHE`](@ref), it is returned immediately. Otherwise, the
model is constructed via [`_leading_residual_terms_any`](@ref), stored in the
cache, and then returned.

The returned tuple contains:

- `ks`: indices of the leading nonzero residual terms,
- `coeffs`: corresponding residual coefficients,
- `center`: centering convention tag.

# Arguments
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary-condition symbol.
- `Nref`: Reference subdivision count passed to the residual-term builder.

# Keyword arguments
- `nterms`: Number of leading nonzero residual terms to collect.
- `kmax`: Maximum moment order scanned while searching for residual terms.
- `real_type = Float64`:
  Scalar type used for residual-coefficient conversion and cache separation.

# Returns
- `Tuple{Vector{Int}, Vector, Symbol}`:
  `(ks, coeffs, center)` for the requested residual model, with `coeffs`
  stored in the active scalar type.

# Notes
- The cache key is `(rule, boundary, nterms, kmax, real_type)`.
- `Nref` is forwarded to the builder when the model is first created, but it is
  not part of the cache key in the current implementation.
"""
function _get_residual_model_fixed(
    rule::Symbol,
    boundary::Symbol,
    Nref::Int;
    nterms::Int,
    kmax::Int,
    real_type = Float64,
)
    T = real_type
    key = (rule, boundary, nterms, kmax, T)

    if haskey(_RES_MODEL_CACHE, key)
        return _RES_MODEL_CACHE[key]
    end

    ks, coeffs, center = _leading_residual_terms_any(
        rule, boundary, Nref;
        nterms = nterms,
        kmax   = kmax,
        real_type = T,
    )

    _RES_MODEL_CACHE[key] = (ks, coeffs, center)
    return ks, coeffs, center
end

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
  resulting coefficients to `Float64` here.
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
    QuadratureUtils._decode_boundary(boundary)

    if NewtonCotes._is_newton_cotes_rule(rule)
        if nterms == 1
            k, coeffR = ErrorNewtonCotesDerivative._leading_midpoint_residual_term(
                rule, boundary, Nsub; kmax = min(kmax, 64)
            )
            return [k], T[convert(T, coeffR)], :mid
        else
            ks, coeffsR = ErrorNewtonCotesDerivative._leading_midpoint_residual_terms(
                rule, boundary, Nsub; nterms = nterms, kmax = kmax
            )
            return ks, T.(coeffsR), :mid
        end
    end

    if Gauss._is_gauss_rule(rule)
        ks, coeffs = ErrorGaussDerivative._leading_midpoint_residual_terms_gauss_float(
            rule, boundary, Nsub; nterms = nterms, kmax = kmax
        )
        return ks, T.(coeffs), :mid
    end

    if BSpline._is_bspline_rule(rule)
        ks, coeffs = ErrorBSplineDerivative._leading_midpoint_residual_terms_bspline_float(
            rule, boundary, Nsub; nterms = nterms, kmax = kmax, λ = 0.0
        )
        return ks, T.(coeffs), :mid
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
        tol_abs = nothing,
        tol_rel = nothing,
        real_type = Float64,
    ) -> Tuple{Vector{Int}, Symbol}

Extract only the residual indices `k` of the first `nterms` nonzero midpoint-shifted
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
- `boundary`: Boundary pattern symbol.
- `Nsub`: Number of unit blocks in the dimensionless tiling domain.

# Keyword arguments
- `nterms`: Number of residual indices to collect.
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
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nterms < 1` or `kmax < 0`.
- Throws if `boundary` is invalid or `rule` is unsupported.
- Propagates backend residual-detection errors.
"""
function _leading_residual_ks_with_center_any(
    rule::Symbol,
    boundary::Symbol,
    Nsub::Int;
    nterms::Int,
    kmax::Int = 256,
    tol_abs = nothing,
    tol_rel = nothing,
    real_type = Float64,
)::Tuple{Vector{Int}, Symbol}

    T = real_type
    tol_abs_T = isnothing(tol_abs) ? T(5e4) * eps(T) : convert(T, tol_abs)
    tol_rel_T = isnothing(tol_rel) ? T(5e4) * eps(T) : convert(T, tol_rel)

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1 (got $nterms)")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0 (got $kmax)")

    QuadratureUtils._decode_boundary(boundary)
    center = :mid

    if NewtonCotes._is_newton_cotes_rule(rule)
        ks = Int[]
        c  = NewtonCotes.RBig(BigInt(Nsub), 2)
        Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)

        β = NewtonCotes._assemble_composite_beta_rational(
            NewtonCotes._parse_newton_p(rule), boundary, Nsub
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
                length(ks) == nterms && return ks, center
            end
        end

        JobLoggerTools.error_benji(
            "Could not collect nterms=$nterms Newton-Cotes residual ks up to kmax=$kmax"
        )
    end

    if Gauss._is_gauss_rule(rule)
        npts = Gauss._parse_gauss_p(rule)

        if boundary === :LU_INEX || boundary === :LU_EXIN
            ks = Int[]
            k0 = 2 * npts - 1
            for j in 0:(nterms - 1)
                push!(ks, k0 + 2 * j)
            end
            return ks, center
        end

        U, W = Gauss._composite_gauss_u_grid(Nsub, npts, boundary)
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
                length(ks) == nterms && return ks, center
            end
        end

        if !isempty(ks) && ks[1] == 0
            JobLoggerTools.error_benji(
                "Gauss residual ks starts with 0 (unstable moment-test). ks=$ks rule=$rule boundary=$boundary Nsub=$Nsub"
            )
        end

        JobLoggerTools.error_benji(
            "Could not collect nterms=$nterms Gauss residual ks up to kmax=$kmax"
        )
    end

    if BSpline._is_bspline_rule(rule)
        ks, center = ErrorBSplineDerivative._leading_residual_ks_with_center_bspline_float(
            rule, boundary, Nsub;
            nterms = nterms,
            kmax = kmax,
            λ = 0.0,
            tol_abs = Float64(tol_abs_T),
            tol_rel = Float64(tol_rel_T)
        )
        return ks, center
    end

    JobLoggerTools.error_benji("Unsupported rule=$rule for residual ks extraction.")
end

# ============================================================
# Unified public API
# ============================================================

include("ErrorDispatchDerivative/error_estimate_derivative_direct_1d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_direct_2d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_direct_3d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_direct_4d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_direct_nd.jl")

"""
    error_estimate_derivative_direct(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        real_type = nothing,
    )

Unified interface for estimating an axis-separable midpoint-residual truncation-error model.

# Function description
This is the public non-threaded dispatcher for the error-estimation layer.

It routes to the matching dimension-specific estimator:

- [`error_estimate_derivative_direct_1d`](@ref) for `dim == 1`
- [`error_estimate_derivative_direct_2d`](@ref) for `dim == 2`
- [`error_estimate_derivative_direct_3d`](@ref) for `dim == 3`
- [`error_estimate_derivative_direct_4d`](@ref) for `dim == 4`
- [`error_estimate_derivative_direct_nd`](@ref) otherwise

All implementations share the same residual-term extraction logic and the same
derivative-backend interface via [`AutoDerivativeDirect.nth_derivative`](@ref).

# Arguments
- `f`: Integrand callable accepting `dim` positional arguments.
- `a`, `b`: Scalar bounds of the hypercube domain.
- `N`: Number of subintervals per axis.
- `dim`: Number of dimensions.
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern symbol.

# Keyword arguments
- `err_method`: Derivative backend selector passed to [`AutoDerivativeDirect.nth_derivative`](@ref).
- `nerr_terms`: Number of nonzero residual terms to include.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and downstream
  derivative-estimator evaluation.

# Returns
- Same return object as the selected dimension-specific estimator.

# Errors
- Propagates errors from the selected estimator.
"""
function error_estimate_derivative_direct(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    if dim == 1
        return error_estimate_derivative_direct_1d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 2
        return error_estimate_derivative_direct_2d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 3
        return error_estimate_derivative_direct_3d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 4
        return error_estimate_derivative_direct_4d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    else
        return error_estimate_derivative_direct_nd(
            f, a, b, N, rule, boundary;
            dim = dim,
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    end
end

include("ErrorDispatchDerivative/error_estimate_derivative_jet_1d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_jet_2d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_jet_3d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_jet_4d.jl")
include("ErrorDispatchDerivative/error_estimate_derivative_jet_nd.jl")

"""
    error_estimate_derivative_jet(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        real_type = nothing,
    )

Dispatch to the jet-based error estimator for the requested dimensionality.

# Function description
This function serves as a dimension-based dispatcher for the jet-oriented
error-estimation pipeline. It selects the appropriate backend according to
`dim`:

- `dim == 1` → [`error_estimate_derivative_jet_1d`](@ref)
- `dim == 2` → [`error_estimate_derivative_jet_2d`](@ref)
- `dim == 3` → [`error_estimate_derivative_jet_3d`](@ref)
- `dim == 4` → [`error_estimate_derivative_jet_4d`](@ref)
- otherwise  → [`error_estimate_derivative_jet_nd`](@ref) with `dim = dim`

Each dispatched routine uses derivative jets internally rather than requesting
scalar derivatives one by one.

# Arguments
- `f`: Integrand or scalar callable to be analyzed.
- `a`: Lower integration bound or lower-domain descriptor.
- `b`: Upper integration bound or upper-domain descriptor.
- `N`: Number of subdivisions.
- `dim`: Problem dimensionality.
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary-condition symbol.

# Keyword arguments
- `err_method::Symbol`:
  Derivative backend selector
  (`:forwarddiff | :taylorseries | :fastdifferentiation | :enzyme`).
- `nerr_terms::Int`:
  Number of residual contributions to retain when constructing the effective
  error estimate.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and downstream
  jet-estimator evaluation.

# Returns
- The return value produced by the selected dimension-specific jet estimator.

# Notes
- This dispatcher does not implement the estimator logic itself; it only routes
  the request to the dimension-appropriate backend.
- For dimensions other than `1`, `2`, `3`, and `4`, the generic
  [`error_estimate_derivative_jet_nd`](@ref) path is used.
- This interface parallels the non-jet error-estimation dispatcher, but is
  specialized for jet-based derivative reuse.
"""
function error_estimate_derivative_jet(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    if dim == 1
        return error_estimate_derivative_jet_1d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 2
        return error_estimate_derivative_jet_2d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 3
        return error_estimate_derivative_jet_3d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    elseif dim == 4
        return error_estimate_derivative_jet_4d(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    else
        return error_estimate_derivative_jet_nd(
            f, a, b, N, rule, boundary;
            dim = dim,
            err_method = err_method,
            nerr_terms = nerr_terms,
            real_type = T,
        )
    end
end

end  # module ErrorDispatchDerivative