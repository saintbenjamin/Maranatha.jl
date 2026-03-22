# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatch.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module QuadratureDispatch

Unified tensor-product quadrature dispatch layer for `Maranatha.Quadrature`.

# Module description
`QuadratureDispatch` exposes the main integration front-end used by the runner
and reporting layers. It combines 1-dimensional nodes and weights from
`QuadratureNodes` into tensor-product quadrature sums and selects between the
available execution backends.

Current responsibilities include:

- providing dimension-specific and generic `quadrature` entry points,
- validating scalar versus axis-wise `rule` / `boundary` specifications,
- dispatching between plain CPU, threaded-subgrid, and CUDA execution paths,
- normalizing interval inputs for scalar and per-axis domains.

# Notes
- This module is the primary quadrature entry point inside the package.
- Rule-family-specific logic remains in the dedicated backend modules
  `NewtonCotes`, `Gauss`, and `BSpline`.
"""
module QuadratureDispatch

import ..JobLoggerTools
import ..QuadratureBoundarySpec
import ..QuadratureRuleSpec
import ..QuadratureNodes

include("internal/_resolve_dispatch_type_and_lambda.jl")
include("internal/_dispatch_local_quadrature.jl")

include("QuadratureDispatchThreadedSubgrid/QuadratureDispatchThreadedSubgrid.jl")
include("QuadratureDispatchCUDA/QuadratureDispatchCUDA.jl")

using .QuadratureDispatchThreadedSubgrid
using .QuadratureDispatchCUDA

include("internal/_dispatch_backend_quadrature.jl")

"""
    quadrature_1d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

Evaluate a ``1``-dimensional quadrature over ``[a,b]``.

# Function description
This routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)` and computes:
```math
\\sum_i w_i f(x_i).
```

Rule-specific validation is centralized in the node/weight generator.
For `dim = 1`, `rule` and `boundary` may be given either as scalar
specifications or as length-1 axis-wise specifications.

# Arguments
- `f`: Integrand callable ``f(x)``.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals / blocks.
- `rule`: Integration rule specification.
- `boundary`: Boundary specification.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

# Errors
- Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_1d(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    λ = nothing,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    b1 = QuadratureBoundarySpec._boundary_at(boundary, 1, 1)

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(
        a,
        b,
        N,
        rule,
        b1;
        λ = λT,
        real_type = T,
    )

    total = zero(T)
    @inbounds for j in eachindex(xs)
        iszero(ws[j]) && continue
        val = f(xs[j])
        total += ws[j] * val
    end
    return total
end

"""
    quadrature_2d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

Evaluate a ``2``-dimensional tensor-product quadrature.

# Function description
This routine supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval ``[a,b]`` is used on both axes.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors, they are interpreted as per-axis bounds,
  and the domain becomes
  ``[a_1,b_1] \\times [a_2,b_2]``.

The routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref) for each active axis and forms
the tensor-product sum
```math
\\sum_i \\sum_j w_i w_j f(x_i, y_j).
```

# Arguments

* `f`: Integrand callable `f(x, y)`.
* `a`, `b`:
  Either scalar bounds used on both axes, or length-2 tuples/vectors specifying
  per-axis rectangular bounds.
* `N`: Number of intervals / blocks per axis.
* `rule`:
  Either a scalar rule symbol shared across both axes, or a length-2
  tuple/vector of per-axis rule symbols.
* `boundary`:
  Either a scalar boundary symbol shared across both axes, or a length-2
  tuple/vector of per-axis boundary symbols.

# Keyword arguments

* `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
* `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns

* `Real`:
  Estimated integral value in the active scalar type.

# Errors

* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 2`
  or `length(b) != 2`.
* Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
* Propagates any error thrown by `f`.
"""
function quadrature_2d(f, a, b, N, rule, boundary; λ = nothing, real_type = nothing)
    return quadrature_nd(f, a, b, N, rule, boundary; dim = 2, λ = λ, real_type = real_type)
end

"""
    quadrature_3d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

Evaluate a ``3``-dimensional tensor-product quadrature.

# Function description
This routine supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval ``[a,b]`` is used on all three axes.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors, they are interpreted as per-axis bounds,
  and the domain becomes
  ``[a_1,b_1] \\times [a_2,b_2] \\times [a_3,b_3]``.

The routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref) for each active axis and forms
the tensor-product sum
```math
\\sum_i \\sum_j \\sum_k w_i w_j w_k f(x_i, y_j, z_k).
```

# Arguments

* `f`: Integrand callable `f(x, y, z)`.
* `a`, `b`:
  Either scalar bounds used on all axes, or length-3 tuples/vectors specifying
  per-axis rectangular bounds.
* `N`: Number of intervals / blocks per axis.
* `rule`:
  Either a scalar rule symbol shared across all axes, or a length-3
  tuple/vector of per-axis rule symbols.
* `boundary`:
  Either a scalar boundary symbol shared across all axes, or a length-3
  tuple/vector of per-axis boundary symbols.

# Keyword arguments

* `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
* `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns

* `Real`:
  Estimated integral value in the active scalar type.

# Errors

* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 3`
  or `length(b) != 3`.
* Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
* Propagates any error thrown by `f`.
"""
function quadrature_3d(f, a, b, N, rule, boundary; λ = nothing, real_type = nothing)
    return quadrature_nd(f, a, b, N, rule, boundary; dim = 3, λ = λ, real_type = real_type)
end

"""
    quadrature_4d(
        f, 
        a, 
        b, 
        N, 
        rule,
        boundary;
        λ = nothing,
        real_type = nothing,
    ) -> Real

Evaluate a ``4``-dimensional tensor-product quadrature.

# Function description
This routine supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the same interval ``[a,b]`` is used on all four axes.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors, they are interpreted as per-axis bounds,
  and the domain becomes
  ``[a_1,b_1] \\times [a_2,b_2] \\times [a_3,b_3] \\times [a_4,b_4]``.

The routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref) for each active axis and forms
the tensor-product sum
```math
\\sum_i \\sum_j \\sum_k \\sum_\\ell w_i w_j w_k w_\\ell f(x_i, y_j, z_k, t_\\ell) \\, .
```

# Arguments

* `f`: Integrand callable `f(x, y, z, t)`.
* `a`, `b`:
  Either scalar bounds used on all axes, or length-4 tuples/vectors specifying
  per-axis rectangular bounds.
* `N`: Number of intervals / blocks per axis.
* `rule`:
  Either a scalar rule symbol shared across all axes, or a length-4
  tuple/vector of per-axis rule symbols.
* `boundary`:
  Either a scalar boundary symbol shared across all axes, or a length-4
  tuple/vector of per-axis boundary symbols.

# Keyword arguments

* `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
* `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns

* `Real`:
  Estimated integral value in the active scalar type.

# Errors

* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 4`
  or `length(b) != 4`.
* Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
* Propagates any error thrown by `f`.
"""
function quadrature_4d(f, a, b, N, rule, boundary; λ = nothing, real_type = nothing)
    return quadrature_nd(f, a, b, N, rule, boundary; dim = 4, λ = λ, real_type = real_type)
end

"""
    quadrature_nd(
        f,
        a,
        b,
        N,
        rule,
        boundary;
        dim::Int,
        λ = nothing,
        real_type = nothing,
    ) -> Real

Perform a general tensor-product quadrature in `dim` dimensions.

# Function description
This routine supports two domain conventions:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, they are interpreted as
  per-axis bounds, and the domain becomes
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

The routine obtains ``1``-dimensional nodes and weights from
[`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref), then enumerates all
tensor-product index tuples with an odometer-style update.

Both `rule` and `boundary` may be supplied either as scalar specifications
shared across all axes or as explicit tuple / vector specifications of length
`dim`.

For each multi-index ``(i_1, \\ldots, i_{\\texttt{dim}})``, it forms the weight
product and evaluates the integrand as
``f(x_1, x_2, \\ldots, x_{\\texttt{dim}})`` using splatting.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`:
  Either scalar bounds defining a hypercube domain, or tuples/vectors of length
  `dim` defining a rectangular per-axis domain.
- `N`: Number of subdivisions / blocks per axis.
- `rule`:
  Either a scalar rule symbol shared across all axes, or a tuple/vector of
  per-axis rule symbols of length `dim`.
- `boundary`:
  Either a scalar boundary symbol shared across all axes, or a tuple/vector of
  per-axis boundary symbols of length `dim`.
- `dim`: Number of dimensions.

# Keyword arguments
- `dim::Int`:
  Number of dimensions.
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `real_type = nothing`:
  Optional scalar type used internally for node/weight construction and
  accumulation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

# Errors
- Throws `ArgumentError` if ``\\texttt{dim} < 1``.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if axis-wise `rule` or `boundary` specifications are
  inconsistent with `dim`.
- Propagates any error thrown by
  [`QuadratureNodes.get_quadrature_1d_nodes_weights`](@ref).
- Propagates any error thrown by `f`.
"""
function quadrature_nd(
    f,
    a,
    b,
    N,
    rule,
    boundary;
    dim::Int,
    λ = nothing,
    real_type = nothing,
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type
    λT = isnothing(λ) ? zero(T) : convert(T, λ)

    QuadratureRuleSpec._validate_rule_spec(rule, dim)
    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)

    xs_list = Vector{Vector{T}}(undef, dim)
    ws_list = Vector{Vector{T}}(undef, dim)

    if !(a isa AbstractVector || a isa Tuple)
        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a,
                b,
                N,
                rule,
                boundary;
                λ = λT,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    else
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))

        for d in 1:dim
            xs_list[d], ws_list[d] = QuadratureNodes.get_quadrature_1d_nodes_weights(
                a[d],
                b[d],
                N,
                rule,
                boundary;
                λ = λT,
                real_type = T,
                axis = d,
                dim = dim,
            )
        end
    end

    idx = ones(Int, dim)
    total = zero(T)
    args = Vector{T}(undef, dim)

    @inbounds while true
        wprod = one(T)

        for d in 1:dim
            i = idx[d]
            args[d] = xs_list[d][i]
            wprod *= ws_list[d][i]
        end

        if !iszero(wprod)
            total += wprod * f(args...)
        end

        d = dim
        while d >= 1
            idx[d] += 1
            if idx[d] <= length(xs_list[d])
                break
            else
                idx[d] = 1
                d -= 1
            end
        end
        d == 0 && break
    end

    return total
end

"""
    quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ = nothing,
        use_cuda::Bool = false,
        threaded_subgrid::Bool = false,
        real_type = nothing,
    ) -> Real

Unified public dispatcher for tensor-product quadrature evaluation.

# Function description
This function provides the main quadrature entry point for `Maranatha`.
It selects one of three execution paths:

- CUDA-based quadrature when `use_cuda == true`,
- CPU threaded-subgrid quadrature when `threaded_subgrid == true`,
- otherwise the dimension-specialized local quadrature path
  ([`quadrature_1d`](@ref), [`quadrature_2d`](@ref),
  [`quadrature_3d`](@ref), [`quadrature_4d`](@ref), or
  [`quadrature_nd`](@ref)).

The optional parameter `λ` is forwarded mainly for smoothing B-spline rules,
and `real_type` controls the internal scalar type used for conversion and
accumulation.

This dispatcher accepts both hypercube-style scalar bounds and axis-wise
rectangular bounds. Support for axis-wise bounds is provided by the selected
backend; on the local CPU path, rectangular domains are supported for
`quadrature_2d`, `quadrature_3d`, `quadrature_4d`, and `quadrature_nd`.

# Arguments
- `integrand`:
  Callable integrand accepting `dim` scalar positional arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds.
- `N`:
  Number of subdivisions or composite blocks per axis.
- `dim`:
  Number of dimensions.
- `rule`:
  Quadrature rule specification. This may be either a scalar rule symbol
  shared across all axes or a tuple/vector of per-axis rule symbols of length
  `dim`.
- `boundary`:
  Boundary specification. This may be either a scalar boundary symbol shared
  across all axes or a tuple/vector of per-axis boundary symbols of length
  `dim`.

# Keyword arguments
- `λ = nothing`:
  Optional smoothing parameter used only for smoothing B-spline rules.
  If `nothing`, zero is used in the active scalar type.
- `use_cuda::Bool = false`:
  If `true`, dispatch to the CUDA quadrature backend.
- `threaded_subgrid::Bool = false`:
  If `true`, dispatch to the CPU threaded-subgrid backend.
  This option is ignored when `use_cuda == true`.
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion and backend
  evaluation.

# Returns
- `Real`:
  Estimated integral value in the active scalar type.

# Errors
- Propagates validation and backend errors from the selected execution path.
- Throws `ArgumentError` indirectly if the selected backend rejects the supplied
  dimensionality, rule configuration, or bound layout.

# Notes
- Backend priority is `use_cuda` first, then `threaded_subgrid`, then the local
  dimension-dispatched CPU implementation.
- This dispatcher performs strategy selection only; it does not implement a
  separate quadrature rule of its own.
"""
function quadrature(
    integrand,
    a,
    b,
    N,
    dim,
    rule,
    boundary;
    λ = nothing,
    use_cuda::Bool = false,
    threaded_subgrid::Bool = false,
    real_type = nothing,
)
    dispatch_state = _resolve_dispatch_type_and_lambda(
        a,
        b,
        λ,
        real_type,
    )

    return _dispatch_backend_quadrature(
        integrand,
        a,
        b,
        N,
        dim,
        rule,
        boundary;
        λ = dispatch_state.λT,
        use_cuda = use_cuda,
        threaded_subgrid = threaded_subgrid,
        real_type = dispatch_state.T,
    )
end

end  # module QuadratureDispatch
