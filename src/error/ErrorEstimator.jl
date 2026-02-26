# ============================================================================
# src/error/ErrorEstimator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorEstimator

using TaylorSeries
using Enzyme
using ForwardDiff
using LinearAlgebra
using ..JobLoggerTools
using ..Integrate
using ..BodeRule_MinOpen_MaxOpen

export estimate_error

# ============================================================
# Internal helpers (must preserve numerical behavior)
# ============================================================

include("ErrorEstimator/nth_derivative.jl")

"""
    _rule_params_for_tensor_error(
        rule::Symbol
    )

Map `rule` to the derivative order `m` and coefficient `C` used by the
tensor-product derivative-based error heuristics in multidimensional error estimators.

# Arguments
- `rule`: Integration rule symbol.

# Returns
- `(m, C)` where:
  - `m::Int` is the derivative order used in the error estimator,
  - `C` is the rule-dependent coefficient (kept as the same literal type
    as the original implementation).

If `rule` is not supported, returns `(0, 0.0)`.
"""
function _rule_params_for_tensor_error(
    rule::Symbol
)
    # IMPORTANT: keep the exact literals/types consistent with the original code.
    if rule == :simpson13_close
        return (4, -1/180)
    elseif rule == :simpson38_close
        return (4, -1/80)
    elseif rule == :bode_close
        return (6, -2/945)
    elseif rule == :simpson13_open
        return (3, -3/8)
    elseif rule == :simpson38_open
        return (4, 14/45)
    elseif rule == :bode_open
        return (6, 1.0)
    else
        return (0, 0.0)
    end
end

# ============================================================
# Boundary-difference models for open-chain rules (1D–4D)
# ============================================================

"""
    _has_boundary_error_model(
        rule::Symbol
    ) -> Bool

Return `true` if `rule` uses a boundary-difference leading-term error model.

# Function description
For some opened composite (endpoint-free) rules, the dominant truncation behavior is
often controlled by **boundary corrections** rather than a purely interior
(midpoint) derivative sample.

This helper identifies the rule symbols for which the error estimators should switch from the default midpoint-based
tensor heuristic to a boundary-difference model.

# Arguments
- `rule::Symbol`: Integration rule symbol.

# Returns
- `Bool`: `true` if a boundary-difference model is defined for `rule`,
  otherwise `false`.

# Notes
- Currently enabled rules:
  - `:simpson13_open`
  - `:bode_open`
- All other rules fall back to [`_rule_params_for_tensor_error`](@ref)`(rule)`-based
  midpoint/tensor heuristics.
"""
@inline function _has_boundary_error_model(
    rule::Symbol
)::Bool
    return (rule == :simpson13_open) || (rule == :bode_open)
end

"""
    _boundary_error_params(
        rule::Symbol
    ) -> (p, K, dord, off)

Return parameters for the boundary-difference **leading-term** error model
associated with `rule`.

# Function description
This routine provides a compact parameterization for boundary-difference error
heuristics used by opened composite rules in error estimators.

The model is expressed in the form
```math
E \\approx \\texttt{K} \\, h^\\texttt{p} \\, ( D_L - D_R ) \\,
```
where
- ``\\displaystyle{h = \\frac{b-a}{N}}``,
- ``D_L  = f^{(\\texttt{m})}(x_L)`` (or an axis-wise derivative in higher dimensions),
- ``D_R = f^{(\\texttt{m})}(x_R)``,

and the evaluation points are placed symmetrically near both ends:
- ``x_L = a + \\texttt{z} \\, h``
- ``x_R = a + ( N - \\texttt{z} ) \\, h``

This parameterization allows multidimensional error estimators to reuse the same boundary logic
by applying the axis-wise boundary difference while integrating over the other
coordinates via the quadrature weights.

# Arguments
- `rule::Symbol`: Integration rule symbol.

# Returns
- `(p, K, m, z)` where:
  - `p::Int`      : leading power of `h` (i.e., the model scales as ``h^\\texttt{p}``),
  - `K::Float64`  : prefactor multiplying the boundary difference,
  - `m::Int`   : derivative order used in the boundary difference,
  - `z::Float64`: offset (in units of `h`) used to define boundary sample points
    ``x_L = a + \\texttt{z} \\, h`` and ``x_R = a + ( N - \\texttt{z} ) \\, h``.

If `rule` is not supported, returns `(0, 0.0, 0, 0.0)`.

# Notes
- This is a **heuristic leading-term model** used to set a stable error *scale*
  for fitting/extrapolation. It is not a rigorous truncation bound.
- The numerical constants are chosen to match the opened composite rule expansions
  used in this project:
  - `:simpson13_open` uses a third-derivative boundary difference with ``h^4``.
  - `:bode_open`     uses a fifth-derivative boundary difference with ``h^6``.
"""
function _boundary_error_params(
    rule::Symbol
)
    if rule == :simpson13_open
        # E ≈ -(3/8) h^4 [ f'''(a+1.5h) - f'''(a+(N-1.5)h) ]
        return (4, -(3.0/8.0), 3, 1.5)
    elseif rule == :bode_open
        # E ≈ -(95/288) h^6 [ f^(5)(a+2.5h) - f^(5)(a+(N-2.5)h) ]
        return (6, -(95.0/288.0), 5, 2.5)
    else
        return (0, 0.0, 0, 0.0)
    end
end

include("ErrorEstimator/estimate_error_1d.jl")
include("ErrorEstimator/estimate_error_2d.jl")
include("ErrorEstimator/estimate_error_3d.jl")
include("ErrorEstimator/estimate_error_4d.jl")
include("ErrorEstimator/estimate_error_nd.jl")

# ============================================================
# Unified public API
# ============================================================

"""
    estimate_error(
        f, 
        a, 
        b, 
        N, 
        dim, 
        rule
    ) -> Float64

Unified interface for estimating integration error in arbitrary dimensions.

# Function description
Dispatches to the corresponding dimension-specific estimator:
- `dim == 1` ``\\rightarrow`` [`estimate_error_1d`](@ref)
- `dim == 2` ``\\rightarrow`` [`estimate_error_2d`](@ref)
- `dim == 3` ``\\rightarrow`` [`estimate_error_3d`](@ref)
- `dim == 4` ``\\rightarrow`` [`estimate_error_4d`](@ref)
- `dim >= 5` ``\\rightarrow`` [`estimate_error_nd`](@ref)

# Arguments
- `f`: Integrand function (expects `dim` positional arguments).
- `a`, `b`: Bounds for each dimension (interpreted as scalar bounds for a hypercube ``[a,b]^\\texttt{dim}``).
- `N`: Number of subdivisions per axis (subject to rule constraints in ``1``-dimensional case; higher-dimensional error estimators reuse the same rule nodes/weights).
- `dim`: Number of dimensions (`Int`).
- `rule`: Integration rule symbol.

# Returns
- A `Float64` multidimensional error estimate.
"""
function estimate_error(
    f, 
    a, 
    b, 
    N, 
    dim, 
    rule
)
    if dim == 1
        return estimate_error_1d(f, a, b, N, rule)
    elseif dim == 2
        return estimate_error_2d(f, a, b, N, rule)
    elseif dim == 3
        return estimate_error_3d(f, a, b, N, rule)
    elseif dim == 4
        return estimate_error_4d(f, a, b, N, rule)
    else
        return estimate_error_nd(f, a, b, N, rule; dim=dim)
    end
end

end  # module ErrorEstimator