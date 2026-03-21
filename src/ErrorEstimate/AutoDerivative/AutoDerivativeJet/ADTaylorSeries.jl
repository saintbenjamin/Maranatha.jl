# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet/ADTaylorSeries.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ADTaylorSeries

TaylorSeries-based backend for derivative-jet construction.

# Module description
This module implements the derivative-jet backend based on `TaylorSeries.jl`
for the automatic-differentiation layer used by `Maranatha.ErrorEstimate`.

It expands the callable around a scalar point and extracts the function value
and successive derivatives up to the requested jet order.

# Notes
- This is an internal backend module.
- Backend selection is handled by `AutoDerivativeJet`.
"""
module ADTaylorSeries

import TaylorSeries

"""
    derivative_jet_taylor(
        f,
        x::Real,
        nmax::Int
    ) -> AbstractVector{<:Real}

Compute the derivative jet of `f` at `x` using `TaylorSeries.jl`.

# Function description
This helper constructs a truncated Taylor polynomial around `x` and extracts
successive derivatives from that representation. The returned vector contains
the function value followed by derivatives of increasing order:

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This backend is often effective when the callable is compatible with
`TaylorSeries.Taylor1` arithmetic and when multiple derivative orders are needed
at the same evaluation point.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `AbstractVector{<:Real}`:
  Derivative jet of length `nmax + 1`, stored in the same scalar type as `x`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates any errors raised by `f` under Taylor-series input.

# Notes
- If `nmax == 0`, the function returns `[f(x)]` as a one-element vector.
- The output is converted to the same scalar type as `x` (not necessarily `Float64`).
"""
function derivative_jet_taylor(
    f,
    x::Real,
    nmax::Int
)
    T = typeof(x)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = convert(T, x)
    nmax == 0 && return T[convert(T, f(x0))]

    t = TaylorSeries.Taylor1(T, nmax)
    y = f(x0 + t)

    ders = Vector{T}(undef, nmax + 1)
    ders[1] = convert(T, TaylorSeries.constant_term(y))

    tmp = y
    for n in 1:nmax
        tmp = TaylorSeries.derivative(tmp)
        ders[n + 1] = convert(T, TaylorSeries.constant_term(tmp))
    end

    return ders
end

end  # module ADTaylorSeries
