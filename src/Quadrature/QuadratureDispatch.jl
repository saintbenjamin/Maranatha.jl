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

Decode a composite boundary pattern symbol into the left/right local rule kinds.

# Function description
This helper maps the global boundary pattern used by the exact-rational
composite Newton-Cotes assembly into the *local* endpoint kinds:

- `:closed` means the local block includes the endpoint node.
- `:opened` means the local block is shifted (open-type block).

Supported boundary patterns are:
- `:LU_ININ` (Left Closed, Right Closed)
- `:LU_EXIN` (Left Opened, Right Closed)
- `:LU_INEX` (Left Closed, Right Opened)
- `:LU_EXEX` (Left Opened, Right Opened)

# Arguments
- `boundary`: Boundary pattern symbol.

# Returns
- `(Ltype, Rtype)`: A tuple of symbols, each either `:closed` or `:opened`.

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `boundary` is not one of the supported symbols.
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

Construct ``1``-dimensional quadrature nodes and weights on ``[a, b]`` for composite Newton-Cotes rules.

# Function description
This function is the public ``1``-dimensional node/weight generator used by the integration dispatchers.

It supports:

## Composite exact-assembly rules `:newton_pK`
If `rule` is recognized as an NS rule, this routine:
1) Parses `p` from `rule`,
2) Builds (or fetches) the coefficient vector `β` for `(p, boundary, N)`,
3) Forms nodes ``\\texttt{xs}_j = a + j \\, h`` for ``j = 0 , \\ldots , N``,
4) Forms weights ``\\texttt{ws}_j = \\beta_j \\, h``, where ``\\displaystyle{h = \\frac{b-a}{N}}``.

The return types are `Vector{Float64}` for both nodes and weights.

# Arguments
- `a`, `b`: Lower/upper bounds of the 1D interval.
- `N`: Number of subintervals (must satisfy ``N \\ge 1`` and composite constraints for the selected boundary).
- `rule`: Rule symbol. Supported:
  - New rules: `:newton_p3`, `:newton_p4`, `:newton_p5`, ...
- `boundary`: Boundary pattern symbol (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
  Required for NS rules.

# Returns
- `xs::Vector{Float64}`: Nodes of length ``N+1``.
- `ws::Vector{Float64}`: Weights of length ``N+1``.

# Errors
- Throws `ArgumentError` if ``N < 1``.
- Throws (via [`Maranatha.Utils.JobLoggerTools.error_benji`](@ref)) if `boundary` is invalid,
  if the composite constraint fails, or if `rule` is unsupported.

# Notes
- This function currently errors on non-NS rules unless you extend the fallback branch
  with your pre-existing implementation.
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

    # --- composite NS branch ---
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

    # --- composite GAUSS branch ---
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

Evaluate a tensor-product Newton-Cotes quadrature on the hypercube ``[a,b]^{\\texttt{dim}}``.

# Function description
This function serves as the unified integration dispatcher within the `Maranatha.jl` pipeline.

1) It builds the **1D nodes and weights** for the selected Newton-Cotes `rule`
   on ``[a,b]`` with resolution `N`.
2) It evaluates the **tensor-product quadrature** in `dim` dimensions by
   enumerating the multi-index over the 1D nodes and accumulating the weighted
   sum of ``\\texttt{integrand}(x_1,\\,\\ldots,\\,x_{\\texttt{dim}})``.

The same bounds ``[a,b]`` are applied along every axis, i.e. the integration domain
is ``[a,b]^{\\texttt{dim}}``.

# Arguments
- `integrand`: A callable that accepts exactly `dim` positional arguments
  (function, closure, or callable struct).
- `a`, `b`: Lower/upper bounds applied to every axis.
- `N`: Number of subintervals per axis (rule-specific constraints apply).
- `dim`: Dimensionality (must satisfy `dim ≥ 1`).
- `rule`: Quadrature rule symbol (e.g. `:simpson13_close`, `:simpson38_open`, `:bode_close`, ...).

# Returns
- `Float64`: Estimated integral value.

# Errors
- Throws an error if `dim < 1`.
- Throws an error if `rule` is unknown or if `N` violates rule-specific constraints.
- Any error thrown by `integrand` during evaluation is propagated.
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