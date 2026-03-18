# ============================================================================
# src/ErrorEstimate/AutoDerivativeJet/ADEnzyme.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ADEnzyme

import Enzyme


"""
    derivative_jet_enzyme(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using repeated `Enzyme` gradients.

# Function description
This helper constructs higher-order derivatives recursively by treating each
previous derivative as a new scalar callable and applying
`Enzyme.gradient(Enzyme.Reverse, ...)` repeatedly. The output vector is

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

This provides a jet-oriented interface for Enzyme-based differentiation inside
the error-estimation subsystem.

# Arguments
- `f`: Scalar callable.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates errors raised by `Enzyme` or by the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- This routine assumes a scalar output and extracts the single gradient entry
  via `only(...)`.
"""
function derivative_jet_enzyme(
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
        g = t -> only(Enzyme.gradient(Enzyme.Reverse, prev, float(t)))
        ders[n + 1] = float(g(x0))
    end

    return ders
end

end  # module ADEnzyme