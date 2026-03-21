# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet/ADForwardDiff.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ADForwardDiff

ForwardDiff-based backend for derivative-jet construction.

# Module description
This module implements the derivative-jet backend based on `ForwardDiff.jl`
for the automatic-differentiation layer used by `Maranatha.ErrorEstimate`.

It constructs the derivative jet
`[f(x), f'(x), ..., f^(nmax)(x)]`
by repeated forward-mode differentiation of nested scalar callables.

# Notes
- This is an internal backend module.
- Backend selection is handled by `AutoDerivativeJet`.
"""
module ADForwardDiff

import ForwardDiff

"""
    derivative_jet_forwarddiff(
        f,
        x::Real,
        nmax::Int
    ) -> AbstractVector{<:Real}

Compute the derivative jet of `f` at `x` using repeated `ForwardDiff`
applications.

# Function description
This helper builds higher-order derivatives recursively by repeatedly applying
`ForwardDiff.derivative` to the previously constructed callable. The resulting
vector has the form

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This backend is simple and general for scalar real-to-real callables that are
compatible with `ForwardDiff`.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `AbstractVector{<:Real}`:
  Derivative jet of length `nmax + 1`, stored in the same scalar type as `x`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates backend failures from `ForwardDiff` or the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- This implementation evaluates derivatives sequentially, so very high orders
  may become expensive or numerically delicate.
"""
function derivative_jet_forwarddiff(
    f,
    x::Real,
    nmax::Int
)
    T = typeof(x)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = convert(T, x)
    ders = Vector{T}(undef, nmax + 1)
    ders[1] = convert(T, f(x0))

    g = f
    for n in 1:nmax
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
        ders[n + 1] = convert(T, g(x0))
    end

    return ders
end

end  # module ADForwardDiff
