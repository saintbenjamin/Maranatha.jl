# ============================================================================
# src/Quadrature/QuadratureNodes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureNodes

One-dimensional quadrature node and weight generator for rule-dispatched
integration backends.

# Module description
`QuadratureNodes` constructs quadrature nodes and weights on a one-dimensional
interval according to a symbolic rule specification. It serves as the central
entry point for all tensor-product quadrature drivers, which build higher-
dimensional integrators by combining these 1D components.

The module dispatches to the appropriate rule family backend, including
composite Newton-Cotes, composite Gauss rules, and spline-based quadrature.

# Responsibility in the quadrature layer

Within the overall architecture:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureNodes` | construct 1D nodes and weights |
| QuadratureDispatch | evaluate multi-dimensional integrals |

Thus, this module defines the geometric and weighting structure of the
quadrature rule, but does not perform the tensor-product accumulation itself.

# Supported rule families

The generator currently supports:

- Newton-Cotes composite rules (`:newton_p*`)
- Gauss-family composite rules (`:gauss_p*`)
- B-spline-based quadrature rules (`:bspline_*`)

Each family may impose its own constraints on the boundary selector or other
parameters.

# Overview

The primary public interface is:

| Function | Responsibility |
|:--|:--|
| [`get_quadrature_1d_nodes_weights`](@ref) | construct nodes and weights on `[a,b]` |

# Notes

- The same nodes and weights may be reused across multiple dimensions in
  tensor-product quadrature.
- Boundary-condition semantics are interpreted via
  [`QuadratureBoundarySpec._decode_boundary`](@ref).
- The module returns floating-point nodes and weights in the active scalar type,
  suitable for numerical integration backends.
"""
module QuadratureNodes

import ..JobLoggerTools
import ..QuadratureBoundarySpec
import ..Quadrature.QuadratureRuleSpec
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline

"""
    get_quadrature_1d_nodes_weights(
        a::Real,
        b::Real,
        N::Int,
        rule,
        boundary;
        λ = nothing,
        real_type = nothing,
        axis::Int = 1,
        dim::Int = 1,
    ) -> (xs, ws)

Construct ``1``-dimensional quadrature nodes and weights on ``[a,b]`` for a
rule-dispatched quadrature backend.

# Function description
This routine is the public node/weight generator used by the tensor-product
quadrature drivers. It dispatches to the appropriate backend according to
`rule`.

The `rule` and `boundary` arguments may be supplied either as scalar
specifications shared across all axes or as axis-wise tuple / vector
specifications. In the axis-wise case, the local rule and boundary used here
are selected by the `axis` / `dim` keywords.

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
these rules are restricted to `boundary = :LU_ININ`. The optional smoothing
parameter `λ` is used only for smoothing B-spline rules.

# Arguments
- `a`, `b`: Lower and upper bounds of the interval.
- `N`: Number of composite blocks / subintervals (``N \\ge 1``).
- `rule`: Quadrature rule specification.
- `boundary`: Boundary specification.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node and weight construction.
- `axis::Int = 1`:
  Axis selector used when resolving axis-wise `rule` / `boundary`
  specifications.
- `dim::Int = 1`:
  Problem dimension used when validating axis-wise `rule` / `boundary`
  specifications.

# Returns
- `xs`: Quadrature nodes on ``[a,b]`` in the active scalar type.
- `ws`: Corresponding quadrature weights in the active scalar type.

# Errors
- Throws `ArgumentError` if ``N < 1``.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification is
  inconsistent with `dim`.
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if the resolved axis-local
  boundary is invalid, if rule-specific constraints fail, or if the resolved
  rule is unsupported.
"""
function get_quadrature_1d_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    rule,
    boundary;
    λ = nothing,
    real_type = nothing,
    axis::Int = 1,
    dim::Int = 1,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    QuadratureRuleSpec._validate_rule_spec(rule, dim)
    rl = QuadratureRuleSpec._rule_at(rule, axis, dim)

    bd = QuadratureBoundarySpec._boundary_at(boundary, axis, dim)
    QuadratureBoundarySpec._decode_boundary(bd)

    if NewtonCotes._is_newton_cotes_rule(rl)
        p = NewtonCotes._parse_newton_p(rl)
        β = NewtonCotes._get_beta(T, p, bd, N)

        aa = convert(T, a)
        bb = convert(T, b)
        h = (bb - aa) / T(N)

        xs = collect(range(aa, bb; length = N + 1))
        ws = Vector{T}(undef, N + 1)
        @inbounds for j in 0:N
            ws[j + 1] = β[j + 1] * h
        end
        return xs, ws
    end

    if Gauss._is_gauss_rule(rl)
        npts = Gauss._parse_gauss_p(rl)
        return Gauss._composite_gauss_nodes_weights(
            a, b, N, npts, bd;
            real_type = T,
            λ = λT,
        )
    end

    if BSpline._is_bspline_rule(rl)
        bd === :LU_ININ || JobLoggerTools.error_benji(
            "B-spline rules currently support only boundary=:LU_ININ (clamped). " *
            "Got boundary=$bd for rule=$rl."
        )

        p = BSpline._parse_bspline_p(rl)
        kind = BSpline._bspline_kind(rl)

        if kind === :interp
            return BSpline.bspline_nodes_weights(
                a, b, N, p, bd;
                kind = :interp,
                real_type = T,
            )
        else
            return BSpline.bspline_nodes_weights(
                a, b, N, p, bd;
                kind = :smooth,
                λ = λT,
                real_type = T,
            )
        end
    end

    JobLoggerTools.error_benji("Unsupported rule=$rl.")
end
end  # module QuadratureNodes
