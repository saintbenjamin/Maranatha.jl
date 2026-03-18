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
composite Newton–Cotes, composite Gauss rules, and spline-based quadrature.

# Responsibility in the quadrature layer

Within the overall architecture:

| Layer | Responsibility |
|:------|:---------------|
| `QuadratureUtils` | shared helpers (e.g., boundary decoding) |
| `QuadratureNodes` | construct 1D nodes and weights |
| QuadratureDispatch | evaluate multi-dimensional integrals |

Thus, this module defines the geometric and weighting structure of the
quadrature rule, but does not perform the tensor-product accumulation itself.

# Supported rule families

The generator currently supports:

- Newton–Cotes composite rules (`:newton_p*`)
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
  [`QuadratureUtils._decode_boundary`](@ref).
- The module returns floating-point nodes and weights suitable for numerical
  integration backends.
"""
module QuadratureNodes

import ..JobLoggerTools
import ..Quadrature.QuadratureUtils
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline

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
    boundary::Symbol;
    λ::Float64 = 0.0
)::Tuple{Vector{Float64}, Vector{Float64}}

    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    # boundary validation early
    QuadratureUtils._decode_boundary(boundary)

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
            return BSpline.bspline_nodes_weights(a, b, N, p, boundary; kind=:smooth, λ=λ)
        end
    end

    # ------------------------------------------------------------
    # FALLBACK: other legacy rules (if any)
    # ------------------------------------------------------------
    JobLoggerTools.error_benji("Unsupported rule=$rule.")
end

end  # module QuadratureNodes