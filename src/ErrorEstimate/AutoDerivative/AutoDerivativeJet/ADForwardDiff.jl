# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet/ADForwardDiff.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ADForwardDiff

import ForwardDiff

"""
    derivative_jet_forwarddiff(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

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
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

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
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)
    ders = Vector{Float64}(undef, nmax + 1)
    ders[1] = float(f(x0))

    g = f
    for n in 1:nmax
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
        ders[n + 1] = float(g(x0))
    end

    return ders
end

end  # module ADForwardDiff