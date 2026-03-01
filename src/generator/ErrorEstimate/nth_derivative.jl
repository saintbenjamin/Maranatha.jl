# ============================================================================
# src/error/ErrorEstimator/nth_derivative.jl
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
This routine evaluates the Taylor expansion of ``f(x + t)`` around ``x``
up to order `n` using a [`TaylorSeries.Taylor1`](https://juliadiff.org/TaylorSeries.jl/stable/api/#TaylorSeries.Taylor1) expansion variable.  
The `n`-th derivative is obtained from the `n`-th Taylor coefficient
multiplied by ``n!``.

Unlike the [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) implementation, this method performs *higher-order
differentiation in a single pass* rather than recursively applying
first derivatives. It is useful for benchmarking alternative AD strategies
and for testing high-order derivative extraction based on truncated
power-series arithmetic.

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
- Internally converts `x` to `Float64` to match the surrounding numeric policy.
- This method may allocate significantly more memory than [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/),
  especially when used inside large loops or with high expansion orders.
"""
@inline function nth_derivative_taylor(f, x::Real, n::Int)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    xx = Float64(x)

    n == 0 && return f(xx)

    t = Taylor1(Float64, n)     # expansion variable (order n)
    y = f(xx + t)               # y is Taylor1 (or compatible)
    return y[n] * factorial(n)  # nth derivative at x
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
        side::Symbol = :mid,
        axis = 0,
        stage::Symbol = :mid
    ) -> Real

Safely compute the `n`-th derivative of scalar callable `g` at point `x`.

# Function description
This wrapper:

1) Attempts to compute the derivative using [`nth_derivative_forwarddiff`](@ref) 
   ([`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)-based).
2) If the result is non-finite, logs a warning and retries using
   [`nth_derivative_taylor`](@ref).
3) If still non-finite, emits a fatal error 
   via [`Maranatha.JobLoggerTools.error_benji`](@ref).

It is designed to be shared across 1D, 2D, 3D, 4D, and general nD
error estimators.

# Keyword arguments
- `h`     : Grid spacing.
- `rule`  : Quadrature rule symbol.
- `N`     : Number of subdivisions.
- `dim`   : Problem dimensionality.
- `side`  : `:L`, `:R`, or `:mid` (boundary location indicator).
- `axis`  : Axis index or symbolic name (for logging).
- `stage` : `:midpoint` or `:boundary` (error-model stage).

# Returns
- The finite `n`-th derivative value ``f^{(n)}(x)``.

# Notes
- This function is marked `@inline` so that Julia can inline it into
  tight quadrature loops without overhead.
"""
@inline function nth_derivative(
    g,
    x,
    n;
    h,
    rule,
    N,
    dim::Int,
    side::Symbol = :mid,
    axis = 0,
    stage::Symbol = :midpoint
)

    d = nth_derivative_forwarddiff(g, x, n)

    if !isfinite(d)
        JobLoggerTools.warn_benji(
            "Non-finite derivative (ForwardDiff.jl); trying TaylorSeries.jl fallback " *
            "h=$h x=$x n=$n rule=$rule N=$N dim=$dim " *
            "side=$side axis=$axis stage=$stage"
        )

        d = nth_derivative_taylor(g, x, n)

        if !isfinite(d)
            JobLoggerTools.error_benji(
                "Non-finite derivative after TaylorSeries.jl fallback: " *
                "h=$h x=$x deriv=$d n=$n rule=$rule N=$N dim=$dim " *
                "side=$side axis=$axis stage=$stage"
            )
        end
    end

    return d
end