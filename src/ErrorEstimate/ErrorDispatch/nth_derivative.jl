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

Compute the `n`-th derivative of a scalar callable `f` at `x`
using a Taylor-series expansion.

# Function description
This routine expands ``f(x + t)``` around the scalar point ``x`` using
`TaylorSeries.Taylor1` up to order ``n``, then extracts the ``n``-th derivative
from the resulting series.

Unlike repeated first-derivative application, this backend performs the
higher-order expansion in a single Taylor-series pass.

# Arguments
- `f`: Scalar-to-scalar callable.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Errors
- Throws `ArgumentError` if `n < 0`.

# Notes
- The expansion center is converted to `Float64`.
- This backend is useful as an alternative high-order derivative path and as a
  comparison point against other AD methods.
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

Compute the `n`-th derivative of a scalar callable `f` at `x`
using repeated `Enzyme.gradient` application.

# Function description
This routine builds a nested closure chain of length `n`. Each layer replaces
the current callable by its first derivative computed through Enzyme reverse-mode
automatic differentiation. The final nested callable is then evaluated at `x`.

# Arguments
- `f`: Scalar-to-scalar callable.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Errors
- No explicit validation is performed here; backend errors from Enzyme are
  propagated if the differentiation chain fails.

# Notes
- Inputs are converted to `Float64`.
- This backend is mainly useful as an experimental or benchmarking path for
  scalar higher-order differentiation.
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

Compute the `n`-th derivative of a scalar callable `f` at `x`
using repeated `ForwardDiff.derivative`.

# Function description
This routine constructs a nested derivative closure chain of length ``n``,
then evaluates the resulting callable at ``x``.

It is intentionally written to accept any Julia callable, not only subtypes of
`Function`, so that closures and callable structs are also supported.

# Arguments
- `f`: Scalar-to-scalar callable.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Errors
- No explicit validation is performed here; any differentiation failure from
  `ForwardDiff` is propagated.

# Notes
- This is the default practical backend in the current error-estimation stack.
- The callable restriction `f::Function` is intentionally avoided.
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

Compute the `n`-th derivative of a scalar callable `f` at `x`
using symbolic differentiation via `FastDifferentiation.jl`.

# Function description
This routine evaluates `f` on a symbolic variable, constructs the symbolic
`n`-th derivative expression, compiles that expression to an executable
function, and evaluates it at `x`.

# Arguments
- `f`: Scalar-to-scalar callable that must accept symbolic `Node` inputs.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value `f^{(n)}(x)`.

# Errors
- Throws `ArgumentError` if `n < 0`.
- Propagates symbolic-construction or compilation errors if `f` is not
  compatible with `FastDifferentiation`.

# Notes
- `n == 0` returns `f(x)`.
- This backend is most appropriate for algebraic integrands that can be traced
  symbolically.
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