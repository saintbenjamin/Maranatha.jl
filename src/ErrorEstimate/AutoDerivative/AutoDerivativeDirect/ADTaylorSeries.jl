# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect/ADTaylorSeries.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ADTaylorSeries

TaylorSeries-based backend for scalar direct derivative evaluation.

# Module description
This module implements the direct `n`-th-derivative backend based on
`TaylorSeries.jl` for the automatic-differentiation layer used by
`Maranatha.ErrorEstimate`.

It provides scalar derivative probes at a single point, which are then reused
by derivative-based residual error estimators.

# Notes
- This is an internal backend module.
- Backend selection is handled by `AutoDerivativeDirect`.
"""
module ADTaylorSeries

import TaylorSeries

"""
    nth_derivative_taylor(
        f,
        x::Real,
        n::Int
    )

Compute the ``n``-th derivative of a scalar callable `f` at `x`
using a Taylor-series expansion via [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/).

# Function description
This routine expands ``f(x + t)`` around the scalar point ``x`` using
[`TaylorSeries.Taylor1`](https://juliadiff.org/TaylorSeries.jl/stable/api/#TaylorSeries.Taylor1)
up to order ``n``, then extracts the ``n``-th derivative
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
- The expansion center is converted to the same scalar type as `x` (not necessarily `Float64`).
- This backend is useful as an alternative high-order derivative path and as a
  comparison point against other AD methods.
"""
@inline function nth_derivative_taylor(
    f, 
    x::Real, 
    n::Int
)
    T = typeof(x)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    n == 0 && return convert(T, f(x))

    x0 = convert(T, x)
    t  = TaylorSeries.Taylor1(T, n)

    y = f(x0 + t)
    return convert(T, TaylorSeries.constant_term(TaylorSeries.derivative(y, n)))
end

end  # module ADTaylorSeries
