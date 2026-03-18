# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet/ADTaylorSeries.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ADTaylorSeries

import TaylorSeries

"""
    derivative_jet_taylor(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

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
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates any errors raised by `f` under Taylor-series input.

# Notes
- If `nmax == 0`, the function returns `[f(x)]` as a one-element vector.
- The output is always converted to `Float64`.
"""
function derivative_jet_taylor(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)
    nmax == 0 && return [float(f(x0))]

    t = TaylorSeries.Taylor1(Float64, nmax)
    y = f(x0 + t)

    ders = Vector{Float64}(undef, nmax + 1)
    ders[1] = float(TaylorSeries.constant_term(y))

    tmp = y
    for n in 1:nmax
        tmp = TaylorSeries.derivative(tmp)
        ders[n + 1] = float(TaylorSeries.constant_term(tmp))
    end

    return ders
end

end  # module ADTaylorSeries