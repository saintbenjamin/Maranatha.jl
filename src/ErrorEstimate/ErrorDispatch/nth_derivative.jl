# ============================================================================
# src/ErrorEstimate/ErrorDispatch/nth_derivative.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    nth_derivative_taylor(
        f,
        x::Real,
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using a Taylor expansion via [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/).

# Function description
This routine evaluates the truncated Taylor expansion of ``f(x + t)`` around
``x`` up to order `n` using a
[`TaylorSeries.Taylor1`](https://juliadiff.org/TaylorSeries.jl/stable/api/#TaylorSeries.Taylor1)
expansion variable.

The ``n``-th derivative is extracted from the Taylor-expanded result by taking
the constant term of the ``n``-th Taylor-series derivative `TaylorSeries.derivative(y, n)`.
(No explicit multiplication by ``n!`` is performed in this function.)

Unlike the [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) implementation, this method performs higher-order
differentiation in a single pass rather than recursively applying first
derivatives. It is useful for benchmarking alternative AD strategies and for
testing high-order derivative extraction based on truncated power-series
arithmetic.

This function accepts any callable object `f`, including:
- ordinary functions,
- anonymous closures,
- callable structs (functors).

# Arguments
- `f`: Scalar-to-scalar callable (`f(x)::Number` expected).
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order (must be nonnegative).

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Notes
- The expansion center is forced to `Float64` (`x0 = float(x)`), so the result
  is computed around a `Float64` point even if `x` was not `Float64`.
- Computational cost and allocation typically grow with the Taylor order `n`,
  since power-series arithmetic stores and propagates coefficients up to degree `n`.
"""
@inline function nth_derivative_taylor(f, x::Real, n::Int)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    n == 0 && return f(x)

    x0 = float(x)                         # force scalar center
    t  = TaylorSeries.Taylor1(Float64, n) # pure Taylor variable (const term = 0)

    y = f(x0 + t)                         # CRITICAL: expand around x0
    return TaylorSeries.constant_term(TaylorSeries.derivative(y, n))
end

"""
    nth_derivative_enzyme(
        f,
        x::Real,
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using repeated reverse-mode differentiation via [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/).

# Function description
This routine constructs a nested closure chain of length `n`, where each step
applies [`Enzyme.gradient`](https://enzyme.mit.edu/index.fcgi/julia/stable/api/#Enzyme.gradient-Union{Tuple{N},%20Tuple{ty_0},%20Tuple{ST},%20Tuple{CS},%20Tuple{StrongZero},%20Tuple{RuntimeActivity},%20Tuple{ErrIfFuncWritten},%20Tuple{ABI},%20Tuple{ReturnPrimal},%20Tuple{F},%20Tuple{ForwardMode{ReturnPrimal,%20ABI,%20ErrIfFuncWritten,%20RuntimeActivity,%20StrongZero},%20F,%20ty_0,%20Vararg{Any,%20N}}}%20where%20{F,%20ReturnPrimal,%20ABI,%20ErrIfFuncWritten,%20RuntimeActivity,%20StrongZero,%20CS,%20ST,%20ty_0,%20N}) in reverse mode to obtain a first derivative.
The resulting callable is then evaluated at `x`.

This mirrors the structure of the [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)-based implementation but replaces
forward-mode differentiation with [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/)'s reverse-mode AD. It is intended
primarily for benchmarking and experimentation with [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/) in scalar
high-order differentiation contexts.

Supported callable types include:
- ordinary functions,
- anonymous closures,
- callable structs (functors).

# Arguments
- `f`: Scalar-to-scalar callable (`f(x)::Number` expected).
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order (must be nonnegative).

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Notes
- Reverse-mode AD is typically advantageous for many-input/one-output problems.
  For repeated scalar higher-order derivatives, performance may be worse than
  [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) due to closure nesting and gradient reconstruction overhead.
- This implementation intentionally preserves the closure-based structure
  for fair benchmarking against other approaches.
- Inputs are converted to `Float64` to match surrounding numeric conventions.
- Provided as a **benchmarking reference implementation**, not as the
  recommended production path in the current codebase.
"""
function nth_derivative_enzyme(
    f,
    x::Real,
    n::Int
)
    g = f
    for _ in 1:n
        prev = g
        g = t -> only(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
        # or: g = t -> first(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
    end
    return g(float(x))
end

"""
    nth_derivative_forwarddiff(
        f, 
        x::Real, 
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using repeated [`ForwardDiff.derivative`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.derivative).

# Function description
This routine is intentionally written to accept any **callable** object `f`,
not only subtypes of `Function`. This includes:
- ordinary functions,
- anonymous closures,
- callable structs (functors) such as preset integrands.

This design is required for compatibility with the integrand registry and
preset-style callable wrappers while preserving [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)-based behavior.

# Arguments
- `f`: Scalar-to-scalar callable (e.g., `f(x)::Number`).
- `x::Real`: Point at which the derivative is evaluated.
- `n::Int`: Derivative order (nonnegative integer).

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Notes
- This implementation constructs a nested closure chain of length `n` and then
  evaluates it at `x`. This intentionally matches the original behavior.
- Type restriction `f::Function` is intentionally avoided because callable
  structs are not subtypes of `Function`, but must be supported.
"""
function nth_derivative_forwarddiff(
    f, 
    x::Real, 
    n::Int
)
    g = f
    for _ in 1:n
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
    end
    return g(x)
end

"""
    nth_derivative_fastdifferentiation(
        f,
        x::Real,
        n::Int
    )

Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
using symbolic differentiation via
[`FastDifferentiation.jl`](https://github.com/brianguenter/FastDifferentiation.jl).

# Function description
This routine constructs a symbolic expression graph of the function `f`
using `FastDifferentiation.Node` objects and performs repeated symbolic
differentiation with respect to the variable.

The resulting symbolic derivative expression is then compiled into a
numerical evaluation function and executed at the point `x`.

# Important requirement
The callable `f` must support evaluation on
`FastDifferentiation.Node` inputs.

In other words, `f(t::Node)` must return a symbolic expression rather than
attempting to perform purely numerical operations.

Typical compatible cases include:

- algebraic expressions
- functions composed of standard mathematical operations
- integrand definitions written generically over `Number`

Functions containing strict type restrictions (e.g. `f(x::Float64)`)
or unsupported control flow may not be traceable.

# Arguments
- `f`: Scalar-to-scalar callable (must accept symbolic `Node` inputs).
- `x::Real`: Point at which the derivative is evaluated.
- `n::Int`: Derivative order (nonnegative integer).

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Notes
- Internally, the symbolic derivative is constructed first and then
  compiled into an executable function using [`FastDifferentiation.make_function`](https://brianguenter.github.io/FastDifferentiation.jl/stable/api/#FastDifferentiation.make_function-Union{Tuple{T},%20Tuple{AbstractArray{T},%20Vararg{AbstractVector{%3C:FastDifferentiation.Node}}}}%20where%20T%3C:FastDifferentiation.Node).
- Clearing the `FastDifferentiation` cache avoids graph reuse issues when
  the routine is called repeatedly with different functions.
"""
function nth_derivative_fastdifferentiation(
    f,
    x::Real,
    n::Int
)
    n >= 0 || throw(ArgumentError("n must be ≥ 0 (got n=$n)"))
    n == 0 && return f(x)

    # Clear global symbolic cache to prevent graph reuse across calls.
    FastDifferentiation.clear_cache()

    # Declare symbolic differentiation variable.
    @variables t

    # Build symbolic expression graph by evaluating f on the symbolic variable.
    expr = f(t)

    # Construct the n-th derivative symbolically.
    dexpr = FastDifferentiation.derivative(expr, ntuple(_ -> t, n)...)

    # Compile symbolic expression into a numerical evaluation function.
    exe = FastDifferentiation.make_function([dexpr], [t])

    # Evaluate the compiled derivative at the requested point.
    return exe(float(x))[1]
end

# """
#     nth_derivative_diffractor(
#         f,
#         x::Real,
#         n::Int
#     )

# Compute the `n`-th derivative of a scalar callable `f` at a scalar point `x`
# using automatic differentiation via
# [`Diffractor.jl`](https://github.com/JuliaDiff/Diffractor.jl).

# # Function description
# This routine computes higher-order derivatives by repeatedly applying
# `AbstractDifferentiation.derivative` with a `DiffractorForwardBackend`.

# The implementation constructs a nested closure chain:

# ```math
# f \\to f^{\\prime} \\to f^{\\prime\\prime} \\to \\ldots f^{(n)}
# ```

# and evaluates the resulting function at the input point `x`.

# This mirrors the design used in the `ForwardDiff` implementation,
# preserving identical calling semantics while replacing the backend
# automatic differentiation engine.

# # Arguments
# - `f`: Scalar-to-scalar callable (e.g. `f(x)::Number`).
# - `x::Real`: Point at which the derivative is evaluated.
# - `n::Int`: Derivative order (nonnegative integer).

# # Returns
# - The `n`-th derivative value ``f^{(n)}(x)``.

# # Notes
# - The callable `f` may be any Julia callable object, including:
#   - ordinary functions,
#   - closures,
#   - callable structs (functors).
# - No restriction to `Function` is imposed, allowing compatibility with
#   integrand registries and preset callable objects.
# """
# function nth_derivative_diffractor(
#     f,
#     x::Real,
#     n::Int
# )

#     n >= 0 || throw(ArgumentError("n must be ≥ 0 (got n=$n)"))

#     if n == 0
#         return f(x)
#     end

#     backend = DiffractorForwardBackend()

#     # Build nested derivative closures.
#     # Each iteration replaces g with its derivative function.
#     g = f
#     for _ in 1:n
#         prev = g
#         g = t -> AbstractDifferentiation.derivative(backend, prev, t)
#     end

#     # Evaluate the final derivative at x.
#     return g(x)

# end

# ============================================================
# Safe derivative wrapper (shared by 1D–nD error estimators)
# ============================================================

"""
    nth_derivative(
        g,
        x,
        n;
        h,
        rule,
        N,
        dim::Int,
        err_method::Symbol = :forwarddiff,
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :midpoint
    ) -> Real

Compute the `n`-th derivative of a scalar callable `g` at point `x`
using a selected differentiation backend.

# Function description
This function is a lightweight backend dispatcher used by the error-estimation
pipeline. It does **not** implement automatic fallback or retry logic; it simply
routes the request according to `err_method`:

- `:forwarddiff`         → [`nth_derivative_forwarddiff`](@ref)
- `:taylorseries`        → [`nth_derivative_taylor`](@ref)
- `:fastdifferentiation` → [`nth_derivative_fastdifferentiation`](@ref)
- `:enzyme`              → [`nth_derivative_enzyme`](@ref)

If an unknown `err_method` is provided, the function aborts via
[`JobLoggerTools.error_benji`](@ref) with a context-rich message.

This interface is shared across 1D/2D/3D/4D and general nD error estimators.

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
- The `n`-th derivative value ``g^{(n)}(x)`` as returned by the selected backend.

# Notes
- This function is marked `@inline` so it can be inlined into tight quadrature
  loops with minimal dispatch overhead.
- Any finiteness checks / fallback policies (if desired) must be implemented
  outside this dispatcher.
"""
@inline function nth_derivative(
    g,
    x,
    n;
    h,
    rule,
    N,
    dim::Int,
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)
    if err_method === :forwarddiff
        return nth_derivative_forwarddiff(g, x, n)

    elseif err_method === :taylorseries
        return nth_derivative_taylor(g, x, n)

    elseif err_method === :fastdifferentiation
        return nth_derivative_fastdifferentiation(g, x, n)

    elseif err_method === :enzyme
        return nth_derivative_enzyme(g, x, n)

    else
        JobLoggerTools.error_benji(
            "Unknown err_method=$err_method (expected :forwarddiff, :taylorseries, :fastdifferentiation, or :enzyme) " *
            "h=$h x=$x n=$n rule=$rule N=$N dim=$dim " *
            "side=$side axis=$axis stage=$stage"
        )
    end
end