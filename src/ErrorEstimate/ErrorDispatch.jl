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

import ..LinearAlgebra
import ..TaylorSeries
import ..Enzyme
import ..ForwardDiff
# import ..Diffractor
import ..FastDifferentiation
import ..FastDifferentiation: @variables

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline
import ..Quadrature.QuadratureDispatch
import ..ErrorEstimate.ErrorNewtonCotes
import ..ErrorEstimate.ErrorGauss
import ..ErrorEstimate.ErrorBSpline

"""
    _RESIDUAL_MODEL_CACHE::Dict{Tuple, Tuple}

Global cache for residual-model data keyed by
`(rule, boundary, nterms, kmax)`.

# Description
This cache stores the tuple returned by
[`_get_residual_model_fixed`](@ref), namely:

- `ks`: leading residual indices,
- `coeffs`: corresponding residual coefficients,
- `center`: centering convention tag.

The purpose of this cache is to avoid recomputing the same residual model
for repeated error-estimation calls using identical quadrature settings.

# Notes
- The cache key intentionally excludes `Nref`, because the fixed residual model
  is currently treated as depending only on `(rule, boundary, nterms, kmax)`.
- Cached values are stored in the exact form returned by
  [`_leading_residual_terms_any`](@ref).
"""
const _RESIDUAL_MODEL_CACHE = Dict{Tuple, Tuple}()

"""
    _NTH_DERIV_CACHE::Dict{Tuple{UInt,Float64,Int,Symbol},Float64}

Global cache for scalar derivative evaluations.

# Description
This cache stores previously computed `n`th-derivative values so that repeated
calls with the same function identity, evaluation point, derivative order, and
backend symbol can reuse an earlier result.

# Key structure
Each key has the form:

- `UInt`: hashed or encoded function identity,
- `Float64`: evaluation point,
- `Int`: derivative order,
- `Symbol`: derivative backend or method tag.

# Notes
- This cache is intended for low-level derivative reuse inside the
  error-estimation workflow.
- Clear it with [`clear_error_estimate_caches!`](@ref) when a fresh run is
  desired.
"""
const _NTH_DERIV_CACHE =
    Dict{Tuple{UInt,Float64,Int,Symbol},Float64}()

"""
    DERIVATIVE_JET_CACHE::Dict{Tuple{Any,Float64,Int,Symbol},Vector{Float64}}

Global cache for derivative jets.

# Description
This cache stores vectors of derivatives evaluated at a fixed point, typically
of the form

```julia
[f(x), f'(x), f''(x), ...]
```

up to a requested maximum order. Reusing a previously computed jet is often
more efficient than recomputing each derivative separately.

# Key structure
Each key has the form:

- `Any`: function identity or callable object,
- `Float64`: evaluation point,
- `Int`: maximum derivative order,
- `Symbol`: derivative backend or method tag.

# Notes
- This cache is especially useful for Taylor-series or automatic-differentiation
  based derivative backends.
- Clear it with [`clear_error_estimate_caches!`](@ref) when needed.
"""
const DERIVATIVE_JET_CACHE =
    Dict{Tuple{Any,Float64,Int,Symbol},Vector{Float64}}()

"""
    clear_error_estimate_caches!() -> Nothing

Clear all global caches used by the error-estimation layer.

# Function description
This helper empties the derivative-value cache, derivative-jet cache, and
residual-model cache, then prints the resulting cache sizes through
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
  - [`_NTH_DERIV_CACHE`](@ref)
  - [`DERIVATIVE_JET_CACHE`](@ref)
  - [`_RESIDUAL_MODEL_CACHE`](@ref)
- Emits diagnostic log lines showing the new cache sizes.

# Notes
- The printed sizes should normally all be zero immediately after this call.
"""
function clear_error_estimate_caches!()
    empty!(_NTH_DERIV_CACHE)
    empty!(DERIVATIVE_JET_CACHE)
    empty!(_RESIDUAL_MODEL_CACHE)
    return nothing
end

"""
    _get_residual_model_fixed(
        rule::Symbol,
        boundary::Symbol,
        Nref::Int;
        nterms::Int,
        kmax::Int
    ) -> Tuple{Vector{Int}, Vector{Float64}, Symbol}

Return a cached residual model for a fixed quadrature configuration.

# Function description
This helper retrieves the leading residual-term model associated with a given
quadrature rule and boundary pattern. If a matching model is already present in
[`_RESIDUAL_MODEL_CACHE`](@ref), it is returned immediately. Otherwise, the
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

# Returns
- `Tuple{Vector{Int}, Vector{Float64}, Symbol}`:
  `(ks, coeffs, center)` for the requested residual model.

# Notes
- The cache key is `(rule, boundary, nterms, kmax)`.
- `Nref` is forwarded to the builder when the model is first created, but it is
  not part of the cache key in the current implementation.
"""
function _get_residual_model_fixed(
    rule::Symbol,
    boundary::Symbol,
    Nref::Int;
    nterms::Int,
    kmax::Int
)
    key = (rule, boundary, nterms, kmax)

    if haskey(_RESIDUAL_MODEL_CACHE, key)
        return _RESIDUAL_MODEL_CACHE[key]
    end

    ks, coeffs, center = _leading_residual_terms_any(
        rule, boundary, Nref;
        nterms = nterms,
        kmax   = kmax
    )

    _RESIDUAL_MODEL_CACHE[key] = (ks, coeffs, center)
    return ks, coeffs, center
end

include("ErrorDispatch/nth_derivative.jl")

"""
    _leading_residual_terms_any(
        rule::Symbol,
        boundary::Symbol,
        Nsub::Int;
        nterms::Int = 1,
        kmax::Int = 128
    ) -> (ks, coeffs_float, center)

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

# Returns
- `ks::Vector{Int}`: Residual indices where a nonzero moment was detected.
- `coeffs_float::Vector{Float64}`: Factorial-scaled residual coefficients.
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
    kmax::Int = 128
)::Tuple{Vector{Int}, Vector{Float64}, Symbol}

    QuadratureDispatch._decode_boundary(boundary)

    if NewtonCotes._is_newton_cotes_rule(rule)
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

    if BSpline._is_bspline_rule(rule)
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
- `tol_abs`: Absolute tolerance for floating-point residual detection.
- `tol_rel`: Relative tolerance for floating-point residual detection.

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
    tol_abs::Float64 = 5e4 * eps(Float64),
    tol_rel::Float64 = 5e4 * eps(Float64)
)::Tuple{Vector{Int}, Symbol}

    (nterms >= 1) || JobLoggerTools.error_benji("nterms must be ≥ 1 (got $nterms)")
    (kmax >= 0)   || JobLoggerTools.error_benji("kmax must be ≥ 0 (got $kmax)")

    QuadratureDispatch._decode_boundary(boundary)
    center = :mid

    # ------------------------------------------------------------
    # Newton-Cotes rules: exact rational test (diff != 0)
    # ------------------------------------------------------------
    if NewtonCotes._is_newton_cotes_rule(rule)
        ks = Int[]
        c  = NewtonCotes.RBig(BigInt(Nsub), 2)
        Nrb = NewtonCotes.RBig(BigInt(Nsub), 1)

        β = NewtonCotes._assemble_composite_beta_rational(NewtonCotes._parse_newton_p(rule), boundary, Nsub)

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

        JobLoggerTools.error_benji("Could not collect nterms=$nterms Newton-Cotes residual ks up to kmax=$kmax")
    end

    # ------------------------------------------------------------
    # Gauss rules: Float64 test with tolerances
    # ------------------------------------------------------------
    if Gauss._is_gauss_rule(rule)
        npts = Gauss._parse_gauss_p(rule)

        if boundary === :LU_INEX || boundary === :LU_EXIN
            # For Gauss–Radau rules (one endpoint included), the leading residual
            # moment indices are known analytically: the quadrature is exact for
            # polynomials up to degree (2npts - 2), so the first non-vanishing
            # centered moment occurs at k = 2*npts - 1.  Subsequent residual
            # moments appear with step 2 (odd powers only).
            #
            # The generic Float64 tolerance-based moment test can misclassify
            # low-order moments (e.g. k=0) due to roundoff and cancellation in
            # the composite rule, which would corrupt the extrapolation model.
            #
            # Therefore we bypass the numerical detection and generate the
            # residual indices directly from the theoretical sequence.
            ks = Int[]
            k0 = 2*npts - 1
            for j in 0:(nterms-1)
                push!(ks, k0 + 2*j)
            end
            return ks, center
        end

        # dimensionless u-grid on [0, Nsub]
        U, W = Gauss._composite_gauss_u_grid(Nsub, npts, boundary)
        c = Float64(Nsub) / 2.0

        ks = Int[]
        for k in 1:kmax
            # k=0 is the constant moment (weight sum). It should be exact by construction,
            # but with Float64 tolerance tests and mixed per-block families (Option A),
            # tiny floating drift can falsely flag k=0 as "nonzero residual".
            # Never use k=0 as a residual index for convergence power inference.

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

        if !isempty(ks) && ks[1] == 0
            JobLoggerTools.error_benji("Gauss residual ks starts with 0 (unstable moment-test). ks=$ks rule=$rule boundary=$boundary Nsub=$Nsub")
        end

        JobLoggerTools.error_benji("Could not collect nterms=$nterms Gauss residual ks up to kmax=$kmax")
    end

    if BSpline._is_bspline_rule(rule)
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

Unified interface for estimating an axis-separable midpoint-residual truncation-error model.

# Function description
This is the public non-threaded dispatcher for the error-estimation layer.

It routes to the matching dimension-specific estimator:

- [`error_estimate_1d`](@ref) for `dim == 1`
- [`error_estimate_2d`](@ref) for `dim == 2`
- [`error_estimate_3d`](@ref) for `dim == 3`
- [`error_estimate_4d`](@ref) for `dim == 4`
- [`error_estimate_nd`](@ref) otherwise

All implementations share the same residual-term extraction logic and the same
derivative-backend interface via [`nth_derivative`](@ref).

# Arguments
- `f`: Integrand callable accepting `dim` positional arguments.
- `a`, `b`: Scalar bounds of the hypercube domain.
- `N`: Number of subintervals per axis.
- `dim`: Number of dimensions.
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern symbol.

# Keyword arguments
- `err_method`: Derivative backend selector passed to [`nth_derivative`](@ref).
- `nerr_terms`: Number of nonzero residual terms to include.

# Returns
- Same return object as the selected dimension-specific estimator.

# Errors
- Propagates errors from the selected estimator.
"""
function error_estimate(
    f, 
    a, 
    b, 
    N, 
    dim, 
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1
)
    if dim == 1
        return error_estimate_1d(
            f, a, b, N, rule, boundary,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 2
        return error_estimate_2d(
            f, a, b, N, rule, boundary,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 3
        return error_estimate_3d(
            f, a, b, N, rule, boundary,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 4
        return error_estimate_4d(
            f, a, b, N, rule, boundary,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    else
        return error_estimate_nd(
            f, a, b, N, rule, boundary;
            dim = dim,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    end
end

"""
    _derivative_values_for_ks(
        g,
        x0,
        ks::AbstractVector{<:Integer};
        h,
        rule,
        N,
        dim::Int,
        err_method::Symbol = :forwarddiff,
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :midpoint
    ) -> Vector{Float64}

Return selected derivative values of `g` at `x0` for the derivative orders
listed in `ks`.

# Function description
This helper computes a derivative jet of `g` up to the maximum order appearing
in `ks`, then extracts only the requested derivative orders and returns them as
a dense `Float64` vector.

More precisely, if

```julia
ks = [k₁, k₂, ..., k_m]
```

then the returned vector is

```julia
[g^(k₁)(x0), g^(k₂)(x0), ..., g^(k_m)(x0)]
```

with each entry obtained from the shared jet produced by
[`derivative_jet`](@ref). This is useful when several specific derivative
orders are needed at the same point, since one jet can serve all of them.

# Arguments
- `g`: Scalar callable.
- `x0`: Evaluation point.
- `ks::AbstractVector{<:Integer}`: Requested derivative orders.

# Keyword arguments
- `h`          : Grid spacing.
- `rule`       : Quadrature rule symbol.
- `N`          : Number of subdivisions.
- `dim::Int`   : Problem dimensionality.
- `err_method` : Backend selector
  (`:forwarddiff | :taylorseries | :fastdifferentiation | :enzyme`).
- `side`       : Boundary-location indicator (`:L`, `:R`, or `:mid`).
- `axis`       : Axis index or symbolic name.
- `stage`      : Stage tag for logging (e.g. `:midpoint` or `:boundary`).

# Returns
- `Vector{Float64}`:
  A vector containing the requested derivative values in the same order as `ks`.

# Notes
- If `ks` is empty, the function returns `Float64[]`.
- The derivative jet is computed only up to `maximum(ks)`.
- Since extraction is performed from a shared jet, this helper is typically more
  efficient than requesting each derivative separately.
"""
@inline function _derivative_values_for_ks(
    g,
    x0,
    ks::AbstractVector{<:Integer};
    h,
    rule,
    N,
    dim::Int,
    err_method::Symbol = :forwarddiff,
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)
    isempty(ks) && return Float64[]

    nmax = maximum(ks)

    jet = derivative_jet(
        g,
        x0,
        nmax;
        h = h,
        rule = rule,
        N = N,
        dim = dim,
        err_method = err_method,
        side = side,
        axis = axis,
        stage = stage,
    )

    vals = Vector{Float64}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        k = ks[i]
        vals[i] = float(jet[k + 1])
    end

    return vals
end

"""
    error_estimate_jet(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1
    )

Dispatch to the jet-based error estimator for the requested dimensionality.

# Function description
This function serves as a dimension-based dispatcher for the jet-oriented
error-estimation pipeline. It selects the appropriate backend according to
`dim`:

- `dim == 1` → [`error_estimate_1d_jet`](@ref)
- `dim == 2` → [`error_estimate_2d_jet`](@ref)
- `dim == 3` → [`error_estimate_3d_jet`](@ref)
- `dim == 4` → [`error_estimate_4d_jet`](@ref)
- otherwise  → [`error_estimate_nd_jet`](@ref) with `dim = dim`

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

# Returns
- The return value produced by the selected dimension-specific jet estimator.

# Notes
- This dispatcher does not implement the estimator logic itself; it only routes
  the request to the dimension-appropriate backend.
- For dimensions other than `1`, `2`, `3`, and `4`, the generic
  [`error_estimate_nd_jet`](@ref) path is used.
- This interface parallels the non-jet error-estimation dispatcher, but is
  specialized for jet-based derivative reuse.
"""
function error_estimate_jet(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1
)
    if dim == 1
        return error_estimate_1d_jet(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 2
        return error_estimate_2d_jet(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 3
        return error_estimate_3d_jet(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    elseif dim == 4
        return error_estimate_4d_jet(
            f, a, b, N, rule, boundary;
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    else
        return error_estimate_nd_jet(
            f, a, b, N, rule, boundary;
            dim = dim,
            err_method = err_method,
            nerr_terms = nerr_terms
        )
    end
end

end  # module ErrorDispatch