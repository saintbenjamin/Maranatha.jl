# ============================================================================
# src/ErrorEstimate/ErrorBSplineRefine.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ErrorBSplineRefine

import ..JobLoggerTools
import ..Quadrature.BSpline

# ============================================================
# Internal helpers
# ============================================================

"""
    _require_bspline_rule(
        rule::Symbol
    ) -> Nothing

Validate that `rule` belongs to the supported B-spline quadrature family.

# Function description
This internal helper checks whether the supplied quadrature-rule symbol is a
B-spline rule recognized by the quadrature layer. It is used as a guard before
calling B-spline-specific parsing or node/weight construction routines.

# Arguments
- `rule::Symbol`:
  Quadrature rule symbol to validate.

# Returns
- `nothing`

# Errors
- Throws (via `JobLoggerTools.error_benji`) if `rule` is not a supported
  B-spline rule.

# Notes
- This helper performs only rule-family validation.
- It does not validate `boundary`, `N`, `dim`, or the smoothing parameter `λ`.
"""
@inline function _require_bspline_rule(
    rule::Symbol
)::Nothing
    BSpline._is_bspline_rule(rule) ||
        JobLoggerTools.error_benji("ErrorBSpline only supports B-spline rules (got rule=$rule)")
    return nothing
end

"""
    _bspline_kind_and_lambda(
        rule::Symbol;
        λ::Float64 = 0.0
    ) -> Tuple{Symbol, Float64}

Resolve the B-spline mode and effective smoothing parameter associated with
`rule`.

# Function description
This internal helper interprets the B-spline rule symbol and converts it into a
normalized `(kind, λeff)` pair suitable for
`BSpline.bspline_nodes_weights`.

Currently supported kinds are:

- `:interp`  : interpolation-type B-spline quadrature, which forces `λeff = 0.0`
- `:smooth`  : smoothing-type B-spline quadrature, which uses the caller-provided
  nonnegative `λ`

# Arguments
- `rule::Symbol`:
  B-spline quadrature rule symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  User-supplied smoothing parameter for smoothing B-spline rules.

# Returns
- `Tuple{Symbol, Float64}`:
  A normalized `(kind, λeff)` pair.

# Errors
- Throws if `rule` is not a valid B-spline rule.
- Throws if a smoothing B-spline rule is requested with `λ < 0`.

# Notes
- Interpolation B-spline rules ignore the supplied `λ` and always return `0.0`
  as the effective smoothing parameter.
- This helper does not construct nodes or weights; it only normalizes rule
  metadata.
"""
@inline function _bspline_kind_and_lambda(
    rule::Symbol;
    λ::Float64 = 0.0
)::Tuple{Symbol, Float64}
    _require_bspline_rule(rule)

    kind = BSpline._bspline_kind(rule)

    if kind === :interp
        return :interp, 0.0
    elseif kind === :smooth
        (λ >= 0.0) || JobLoggerTools.error_benji("λ must be ≥ 0 for smoothing B-spline rules (got λ=$λ)")
        return :smooth, λ
    else
        JobLoggerTools.error_benji("Unknown B-spline kind extracted from rule=$rule")
    end
end

"""
    _quadrature_1d_bspline(
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Construct the 1D B-spline quadrature nodes and weights for the interval
`[a, b]`.

# Function description
This helper validates the B-spline rule family, parses the rule order, resolves
the B-spline kind and effective smoothing parameter, and then calls
`BSpline.bspline_nodes_weights` to generate the quadrature nodes and weights
used by the refinement-based error estimator.

# Arguments
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Number of subintervals or composite blocks.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol passed through to the B-spline backend.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter used only for smoothing B-spline rules.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`:
  The quadrature node vector `xs` and corresponding weight vector `ws`.

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Throws if `N < 1`.
- Propagates errors from the underlying B-spline node/weight constructor.

# Notes
- This function only builds the 1D quadrature grid and does not evaluate the
  integrand.
- The returned vectors are intended to be reused in tensor-product evaluations.
"""
@inline function _quadrature_1d_bspline(
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Tuple{Vector{Float64}, Vector{Float64}}

    _require_bspline_rule(rule)
    (N >= 1) || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")

    p = BSpline._parse_bspline_p(rule)
    kind, λeff = _bspline_kind_and_lambda(rule; λ=λ)

    xs, ws = BSpline.bspline_nodes_weights(
        a, b, N, p, boundary;
        kind = kind,
        λ = λeff,
    )

    return xs, ws
end

"""
    _quadrature_value_1d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Evaluate the 1D B-spline quadrature approximation of `f` on `[a, b]`.

# Function description
This helper constructs the 1D B-spline quadrature nodes and weights and then
computes the weighted sum

```julia
\\sum_i w_i f(x_i)
```

as a `Float64` quadrature value.

# Arguments
- `f`:
  Scalar integrand callable accepting one positional argument.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Number of composite blocks.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The 1D B-spline quadrature value.

# Errors
- Propagates validation and backend errors from
  [`_quadrature_1d_bspline`](@ref).

# Notes
- This helper is used internally by the refinement-based error estimator.
- No derivative information is used.
"""
function _quadrature_value_1d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    xs, ws = _quadrature_1d_bspline(a, b, N, rule, boundary; λ=λ)

    acc = 0.0
    @inbounds for i in eachindex(xs)
        acc += ws[i] * f(xs[i])
    end
    return acc
end

"""
    _quadrature_value_2d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Evaluate the 2D tensor-product B-spline quadrature approximation of `f` on the
hypercube `[a, b]^2`.

# Function description
This helper reuses the same 1D B-spline nodes and weights along both axes and
computes the tensor-product quadrature sum

```julia
\\sum_{i,j} w_i w_j f(x_i, y_j).
```

# Arguments
- `f`:
  Scalar integrand callable accepting two positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The 2D tensor-product B-spline quadrature value.

# Errors
- Propagates validation and backend errors from
  [`_quadrature_1d_bspline`](@ref).

# Notes
- The same 1D quadrature grid is used for both axes.
- This routine is specialized for the 2D case to avoid the overhead of the
  generic `nd` recursion path.
"""
function _quadrature_value_2d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    xs, ws = _quadrature_1d_bspline(a, b, N, rule, boundary; λ=λ)

    acc = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wx = ws[i]
        for j in eachindex(xs)
            y = xs[j]
            wy = ws[j]
            acc += (wx * wy) * f(x, y)
        end
    end
    return acc
end

"""
    _quadrature_value_3d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Evaluate the 3D tensor-product B-spline quadrature approximation of `f` on the
hypercube `[a, b]^3`.

# Function description
This helper forms the 3D tensor-product quadrature sum using a shared 1D
B-spline node/weight set on each axis:

```julia
\\sum_{i,j,k} w_i w_j w_k f(x_i, y_j, z_k).
```

# Arguments
- `f`:
  Scalar integrand callable accepting three positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The 3D tensor-product B-spline quadrature value.

# Errors
- Propagates validation and backend errors from
  [`_quadrature_1d_bspline`](@ref).

# Notes
- This specialized implementation avoids the generic recursive `nd` evaluator
  for the common 3D case.
"""
function _quadrature_value_3d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    xs, ws = _quadrature_1d_bspline(a, b, N, rule, boundary; λ=λ)

    acc = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wx = ws[i]
        for j in eachindex(xs)
            y = xs[j]
            wy = ws[j]
            for k in eachindex(xs)
                z = xs[k]
                wz = ws[k]
                acc += (wx * wy * wz) * f(x, y, z)
            end
        end
    end
    return acc
end

"""
    _quadrature_value_4d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Evaluate the 4D tensor-product B-spline quadrature approximation of `f` on the
hypercube `[a, b]^4`.

# Function description
This helper forms the 4D tensor-product quadrature sum using a shared 1D
B-spline node/weight set on each axis:

```julia
\\sum_{i,j,k,l} w_i w_j w_k w_l f(x_i, y_j, z_k, t_l).
```

# Arguments
- `f`:
  Scalar integrand callable accepting four positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The 4D tensor-product B-spline quadrature value.

# Errors
- Propagates validation and backend errors from
  [`_quadrature_1d_bspline`](@ref).

# Notes
- This specialized implementation is intended for the common low-dimensional
  cases where explicit loops are clearer and often faster than the generic
  recursive path.
"""
function _quadrature_value_4d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    xs, ws = _quadrature_1d_bspline(a, b, N, rule, boundary; λ=λ)

    acc = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        wx = ws[i]
        for j in eachindex(xs)
            y = xs[j]
            wy = ws[j]
            for k in eachindex(xs)
                z = xs[k]
                wz = ws[k]
                for l in eachindex(xs)
                    t = xs[l]
                    wt = ws[l]
                    acc += (wx * wy * wz * wt) * f(x, y, z, t)
                end
            end
        end
    end
    return acc
end

"""
    _quadrature_value_nd_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Evaluate the `dim`-dimensional tensor-product B-spline quadrature approximation
of `f` on `[a, b]^dim`.

# Function description
This helper constructs a shared 1D B-spline quadrature grid and applies it to
all axes through a recursive tensor-product traversal. It is used as the
generic fallback for dimensions other than the specialized `1d`, `2d`, `3d`,
and `4d` paths.

# Arguments
- `f`:
  Scalar integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The `dim`-dimensional tensor-product B-spline quadrature value.

# Errors
- Throws if `dim < 1`.
- Propagates validation and backend errors from
  [`_quadrature_1d_bspline`](@ref).

# Notes
- This routine uses a recursive accumulator over a mutable argument buffer.
- It is intended as a general fallback and may be slower than the specialized
  low-dimensional implementations.
"""
function _quadrature_value_nd_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")

    xs, ws = _quadrature_1d_bspline(a, b, N, rule, boundary; λ=λ)

    args = Vector{Float64}(undef, dim)

    function _recur(level::Int, wprod::Float64)::Float64
        if level > dim
            return wprod * f(args...)
        end

        acc = 0.0
        @inbounds for i in eachindex(xs)
            args[level] = xs[i]
            acc += _recur(level + 1, wprod * ws[i])
        end
        return acc
    end

    return _recur(1, 1.0)
end

"""
    _quadrature_value_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    ) -> Float64

Dispatch to the appropriate dimension-specific B-spline quadrature evaluator.

# Function description
This helper selects the specialized B-spline quadrature evaluator matching
`dim`:

- `dim == 1` → [`_quadrature_value_1d_bspline`](@ref)
- `dim == 2` → [`_quadrature_value_2d_bspline`](@ref)
- `dim == 3` → [`_quadrature_value_3d_bspline`](@ref)
- `dim == 4` → [`_quadrature_value_4d_bspline`](@ref)
- otherwise  → [`_quadrature_value_nd_bspline`](@ref)

# Arguments
- `f`:
  Scalar integrand callable.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Number of composite blocks per axis.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `Float64`:
  The quadrature value produced by the selected evaluator.

# Errors
- Propagates errors from the selected dimension-specific routine.

# Notes
- This function only dispatches; it does not implement a separate quadrature
  algorithm.
"""
@inline function _quadrature_value_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)::Float64
    if dim == 1
        return _quadrature_value_1d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 2
        return _quadrature_value_2d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 3
        return _quadrature_value_3d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 4
        return _quadrature_value_4d_bspline(f, a, b, N, rule, boundary; λ=λ)
    else
        return _quadrature_value_nd_bspline(f, a, b, N, dim, rule, boundary; λ=λ)
    end
end

"""
    _estimate_by_refinement_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        dim::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the B-spline quadrature error by comparing coarse and refined
composite quadrature evaluations.

# Function description
This internal helper implements the refinement-difference error estimator for
B-spline quadrature rules. It computes

- a coarse estimate using `N` composite blocks, and
- a refined estimate using `2N` composite blocks,

then forms their difference

```julia
diff = q_fine - q_coarse.
```

The returned named tuple stores both quadrature values, the corresponding mesh
sizes, and the absolute refinement difference used as the effective error
estimate.

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Coarse subdivision count.
- `dim::Int`:
  Number of dimensions.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- `NamedTuple` with fields:
  - `method`      : method tag `:bspline_refinement_difference`
  - `rule`        : quadrature rule symbol
  - `boundary`    : boundary-condition symbol
  - `N_coarse`    : coarse subdivision count
  - `N_fine`      : refined subdivision count (`2N`)
  - `dim`         : dimensionality
  - `h_coarse`    : coarse mesh size
  - `h_fine`      : refined mesh size
  - `q_coarse`    : coarse quadrature value
  - `q_fine`      : refined quadrature value
  - `estimate`    : absolute refinement difference
  - `signed_diff` : signed refinement difference
  - `reference`   : refined quadrature value used as the internal reference

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Throws if `N < 1`.
- Throws if `dim < 1`.
- Propagates errors from the quadrature-evaluation layer.

# Notes
- This estimator does not use derivatives or residual moments.
- The returned `estimate` is currently `abs(q_fine - q_coarse)` without an
  additional Richardson-style normalization factor.
"""
function _estimate_by_refinement_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    dim::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0
)
    _require_bspline_rule(rule)
    (N >= 1) || JobLoggerTools.error_benji("Need N ≥ 1 (got N=$N)")
    (dim >= 1) || JobLoggerTools.error_benji("dim must be ≥ 1 (got dim=$dim)")

    aa = float(a)
    bb = float(b)
    h_coarse = (bb - aa) / N
    h_fine   = (bb - aa) / (2N)

    q_coarse = _quadrature_value_bspline(f, aa, bb, N,  dim, rule, boundary; λ=λ)
    q_fine   = _quadrature_value_bspline(f, aa, bb, 2N, dim, rule, boundary; λ=λ)

    diff = q_fine - q_coarse

    return (;
        method      = :bspline_refinement_difference,
        rule        = rule,
        boundary    = boundary,
        N_coarse    = N,
        N_fine      = 2N,
        dim         = dim,
        h_coarse    = h_coarse,
        h_fine      = h_fine,
        q_coarse    = q_coarse,
        q_fine      = q_fine,
        estimate    = abs(diff),
        signed_diff = diff,
        reference   = q_fine,
    )
end

"""
    error_estimate_1d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the 1D B-spline quadrature error by refinement.

# Function description
This public helper is the 1D specialization of the B-spline refinement-based
error-estimation interface. It forwards the request to
[`_estimate_by_refinement_bspline`](@ref) with `dim = 1`.

# Arguments
- `f`:
  Scalar integrand callable accepting one positional argument.
- `a::Real`:
  Lower integration bound.
- `b::Real`:
  Upper integration bound.
- `N::Int`:
  Coarse subdivision count.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_bspline`](@ref),
  specialized to `dim = 1`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_1d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
)
    return _estimate_by_refinement_bspline(
        f, a, b, N, 1, rule, boundary; λ=λ
    )
end

"""
    error_estimate_2d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the 2D B-spline quadrature error by refinement.

# Function description
This public helper is the 2D specialization of the B-spline refinement-based
error-estimation interface. It forwards the request to
[`_estimate_by_refinement_bspline`](@ref) with `dim = 2`.

# Arguments
- `f`:
  Scalar integrand callable accepting two positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_bspline`](@ref),
  specialized to `dim = 2`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_2d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
)
    return _estimate_by_refinement_bspline(
        f, a, b, N, 2, rule, boundary; λ=λ
    )
end

"""
    error_estimate_3d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the 3D B-spline quadrature error by refinement.

# Function description
This public helper is the 3D specialization of the B-spline refinement-based
error-estimation interface. It forwards the request to
[`_estimate_by_refinement_bspline`](@ref) with `dim = 3`.

# Arguments
- `f`:
  Scalar integrand callable accepting three positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_bspline`](@ref),
  specialized to `dim = 3`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_3d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
)
    return _estimate_by_refinement_bspline(
        f, a, b, N, 3, rule, boundary; λ=λ
    )
end

"""
    error_estimate_4d_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        λ::Float64 = 0.0
    )

Estimate the 4D B-spline quadrature error by refinement.

# Function description
This public helper is the 4D specialization of the B-spline refinement-based
error-estimation interface. It forwards the request to
[`_estimate_by_refinement_bspline`](@ref) with `dim = 4`.

# Arguments
- `f`:
  Scalar integrand callable accepting four positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_bspline`](@ref),
  specialized to `dim = 4`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_4d_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    λ::Float64 = 0.0,
)
    return _estimate_by_refinement_bspline(
        f, a, b, N, 4, rule, boundary; λ=λ
    )
end

"""
    error_estimate_nd_bspline(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        dim::Int,
        λ::Float64 = 0.0
    )

Estimate the `dim`-dimensional B-spline quadrature error by refinement.

# Function description
This public helper is the generic `nd` specialization of the B-spline
refinement-based error-estimation interface. It forwards the request to
[`_estimate_by_refinement_bspline`](@ref) with the user-supplied `dim`.

# Arguments
- `f`:
  Scalar integrand callable accepting `dim` positional arguments.
- `a::Real`:
  Lower integration bound on each axis.
- `b::Real`:
  Upper integration bound on each axis.
- `N::Int`:
  Coarse subdivision count per axis.
- `rule::Symbol`:
  B-spline quadrature rule symbol.
- `boundary::Symbol`:
  Boundary-condition symbol.

# Keyword arguments
- `dim::Int`:
  Problem dimensionality.
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- Same named tuple returned by [`_estimate_by_refinement_bspline`](@ref),
  specialized to the requested `dim`.

# Errors
- Propagates validation and quadrature-evaluation errors from the internal
  refinement estimator.
"""
function error_estimate_nd_bspline(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int,
    λ::Float64 = 0.0,
)
    return _estimate_by_refinement_bspline(
        f, a, b, N, dim, rule, boundary; λ=λ
    )
end

"""
    error_estimate_bspline(
        f,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ::Float64 = 0.0
    )

Unified public dispatcher for B-spline refinement-based error estimation.

# Function description
This function provides the main B-spline-specific entry point for the
refinement-based error-estimation layer. It dispatches to the dimension-specific
specializations:

- `dim == 1` → [`error_estimate_1d_bspline`](@ref)
- `dim == 2` → [`error_estimate_2d_bspline`](@ref)
- `dim == 3` → [`error_estimate_3d_bspline`](@ref)
- `dim == 4` → [`error_estimate_4d_bspline`](@ref)
- otherwise  → [`error_estimate_nd_bspline`](@ref)

# Arguments
- `f`:
  Integrand callable accepting `dim` positional arguments.
- `a`:
  Lower integration bound.
- `b`:
  Upper integration bound.
- `N`:
  Coarse subdivision count.
- `dim`:
  Number of dimensions.
- `rule`:
  B-spline quadrature rule symbol.
- `boundary`:
  Boundary-condition symbol.

# Keyword arguments
- `λ::Float64 = 0.0`:
  Smoothing parameter for smoothing B-spline rules.

# Returns
- The named tuple produced by the selected dimension-specific refinement
  estimator.

# Errors
- Throws if `rule` is not a supported B-spline rule.
- Propagates errors from the selected dimension-specific routine.

# Notes
- This dispatcher is intended for refinement-based error estimation only.
- Unlike the derivative-based error estimators, this interface does not depend
  on a derivative backend or jet construction.
"""
function error_estimate_bspline(
    f,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ::Float64 = 0.0,
)
    _require_bspline_rule(rule)

    if dim == 1
        return error_estimate_1d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 2
        return error_estimate_2d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 3
        return error_estimate_3d_bspline(f, a, b, N, rule, boundary; λ=λ)
    elseif dim == 4
        return error_estimate_4d_bspline(f, a, b, N, rule, boundary; λ=λ)
    else
        return error_estimate_nd_bspline(
            f, a, b, N, rule, boundary;
            dim=dim, λ=λ
        )
    end
end

end  # module ErrorBSplineRefine