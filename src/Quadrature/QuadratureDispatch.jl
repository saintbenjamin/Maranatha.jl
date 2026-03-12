# ============================================================================
# src/Quadrature/QuadratureDispatch.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module QuadratureDispatch

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline

"""
    _decode_boundary(
        boundary::Symbol
    ) -> Tuple{Symbol,Symbol}

Decode a composite boundary selector into left/right local endpoint kinds.

# Function description
This helper maps the global boundary pattern into a pair of local endpoint tags
used by the Newton-Cotes composite assembly:

- `:closed` means the local block includes the endpoint node.
- `:opened` means the local block uses the shifted open-type construction.

Supported patterns are:

- `:LU_ININ` -> `(:closed, :closed)`
- `:LU_EXIN` -> `(:opened, :closed)`
- `:LU_INEX` -> `(:closed, :opened)`
- `:LU_EXEX` -> `(:opened, :opened)`

# Arguments
- `boundary`: Boundary pattern symbol.

# Returns
- `Tuple{Symbol,Symbol}`: `(Ltype, Rtype)`, each equal to `:closed` or `:opened`.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `boundary` is not one of
  `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, or `:LU_EXEX`.
"""
@inline function _decode_boundary(
    boundary::Symbol
)
    if boundary === :LU_ININ
        return (:closed, :closed)
    elseif boundary === :LU_EXIN
        return (:opened, :closed)
    elseif boundary === :LU_INEX
        return (:closed, :opened)
    elseif boundary === :LU_EXEX
        return (:opened, :opened)
    else
        JobLoggerTools.error_benji("boundary must be one of: :LU_ININ | :LU_EXIN | :LU_INEX | :LU_EXEX (got $boundary)")
    end
end

"""
    get_quadrature_1d_nodes_weights(
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    ) -> (xs, ws)

Construct ``1``-dimensional quadrature nodes and weights on ``[a,b]`` for a
rule-dispatched quadrature backend.

# Function description
This routine is the public node/weight generator used by the tensor-product
quadrature drivers. It dispatches to the appropriate backend according to
`rule`.

Supported rule families:

- `:newton_p*` -> exact-rational composite Newton-Cotes assembly
- `:gauss_p*` -> composite Gauss-family rules
- `:bspline_*` -> spline-based quadrature rules

For Newton-Cotes rules, the routine parses `p`, retrieves the composite
coefficient vector ``\\beta``, generates uniform nodes on ``[a,b]``, and forms
weights ``w_j = \\beta_j h`` with ``h = \\dfrac{b-a}{N}``.

For Gauss-family rules, the routine delegates to the composite Gauss backend,
which applies the requested family blockwise.

For B-spline rules, the routine delegates to the spline backend. At present,
these rules are restricted to `boundary = :LU_ININ`.

# Arguments
- `a`, `b`: Lower and upper bounds of the interval.
- `N`: Number of composite blocks / subintervals (``N \\ge 1``).
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern selector.

# Returns
- `xs::Vector{Float64}`: Quadrature nodes on ``[a,b]``.
- `ws::Vector{Float64}`: Corresponding quadrature weights.

# Errors
- Throws `ArgumentError` if ``N < 1``.
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if the boundary is invalid,
  if rule-specific constraints fail, or if `rule` is unsupported.
"""
function get_quadrature_1d_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol
)::Tuple{Vector{Float64}, Vector{Float64}}

    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    # boundary validation early
    _decode_boundary(boundary)

    # --- composite Newton-Cotes branch ---
    if NewtonCotes._is_newton_cotes_rule(rule)
        p = NewtonCotes._parse_newton_p(rule)
        β = NewtonCotes._get_beta_float(p, boundary, N)

        aa = Float64(a)
        bb = Float64(b)
        h = (bb - aa) / Float64(N)

        xs = collect(range(aa, bb; length=N+1))
        ws = Vector{Float64}(undef, N+1)
        @inbounds for j in 0:N
            ws[j+1] = β[j+1] * h
        end
        return xs, ws
    end

    # --- composite Gauss branch ---
    if Gauss._is_gauss_rule(rule)
        npts = Gauss._parse_gauss_p(rule)  # points per block
        return Gauss._composite_gauss_nodes_weights(a, b, N, npts, boundary)
    end

    # --- composite B-SPLINE branch ---
    if BSpline._is_bspline_rule(rule)
        # Policy: B-spline rules support only clamped boundary for now
        if boundary !== :LU_ININ
            JobLoggerTools.error_benji(
                "B-spline rules currently support only boundary=:LU_ININ (clamped). " *
                "Got boundary=$boundary for rule=$rule."
            )
        end

        p = BSpline._parse_bspline_p(rule)
        kind = BSpline._bspline_kind(rule)  # :interp or :smooth

        # NOTE: smoothing λ is fixed to 0.0 for now (pure interpolation-like),
        #       will be wired as a user option later.
        if kind === :interp
            return BSpline.bspline_nodes_weights(a, b, N, p, boundary; kind=:interp)
        else
            return BSpline.bspline_nodes_weights(a, b, N, p, boundary; kind=:smooth, λ=0.0)
        end
    end

    # ------------------------------------------------------------
    # FALLBACK: other legacy rules (if any)
    # ------------------------------------------------------------
    JobLoggerTools.error_benji("Unsupported rule=$rule.")
end

include("QuadratureDispatch/quadrature_1d.jl")
include("QuadratureDispatch/quadrature_2d.jl")
include("QuadratureDispatch/quadrature_3d.jl")
include("QuadratureDispatch/quadrature_4d.jl")
include("QuadratureDispatch/quadrature_nd.jl")

"""
    quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary
    ) -> Float64

Evaluate a tensor-product quadrature on the hypercube ``[a,b]^{\\texttt{dim}}``.

# Function description
This is the unified integration dispatcher for the quadrature layer.

It first builds the underlying ``1``-dimensional nodes and weights through
[`get_quadrature_1d_nodes_weights`](@ref), then chooses a dimension-specific
tensor-product evaluator:

- [`quadrature_1d`](@ref) for `dim == 1`
- [`quadrature_2d`](@ref) for `dim == 2`
- [`quadrature_3d`](@ref) for `dim == 3`
- [`quadrature_4d`](@ref) for `dim == 4`
- [`quadrature_nd`](@ref) otherwise

All axes use the same interval ``[a,b]``, so the integration domain is the
hypercube ``[a,b]^{\\texttt{dim}}``.

# Arguments
- `integrand`: Callable accepting exactly `dim` positional arguments.
- `a`, `b`: Lower and upper bounds applied to every axis.
- `N`: Number of subintervals / blocks per axis.
- `dim`: Number of dimensions.
- `rule`: Quadrature rule symbol.
- `boundary`: Boundary pattern selector.

# Returns
- `Float64`: Estimated integral value.

# Errors
- Throws an error if ``\\texttt{dim} < 1``.
- Throws any rule-validation or backend error propagated from the selected
  quadrature generator.
- Propagates any error thrown by `integrand`.
"""
function quadrature(
    integrand, 
    a, 
    b, 
    N, 
    dim, 
    rule,
    boundary
)
    if dim == 1
        return quadrature_1d(integrand, a, b, N, rule, boundary)
    elseif dim == 2
        return quadrature_2d(integrand, a, b, N, rule, boundary)
    elseif dim == 3
        return quadrature_3d(integrand, a, b, N, rule, boundary)
    elseif dim == 4
        return quadrature_4d(integrand, a, b, N, rule, boundary)
    else
        return quadrature_nd(integrand, a, b, N, rule, boundary; dim=dim)
    end
end

end  # module QuadratureDispatch