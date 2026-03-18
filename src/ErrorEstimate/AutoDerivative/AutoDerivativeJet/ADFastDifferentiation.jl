# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect/ADFastDifferentiation.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ADFastDifferentiation

import FastDifferentiation
import FastDifferentiation: @variables

"""
    derivative_jet_fastdifferentiation(
        f,
        x::Real,
        nmax::Int
    ) -> Vector{Float64}

Compute the derivative jet of `f` at `x` using `FastDifferentiation.jl`.

# Function description
This helper constructs a symbolic differentiation graph for `f(t)` and then
builds all derivatives from order `0` through `nmax` with respect to the same
symbolic variable. The compiled FastDifferentiation function is then evaluated
at `x`, and the result is returned as a `Float64` vector.

The output vector is

```julia
[f(x), f'(x), f''(x), ..., f^(nmax)(x)]
```

# Arguments
- `f`: Scalar callable compatible with FastDifferentiation symbolic variables.
- `x::Real`: Evaluation point.
- `nmax::Int`: Maximum derivative order to compute.

# Returns
- `Vector{Float64}`: Derivative jet of length `nmax + 1`.

# Errors
- Throws `ArgumentError` if `nmax < 0`.
- Propagates symbolic-construction or execution errors from
  `FastDifferentiation` or the callable `f`.

# Notes
- If `nmax == 0`, the function returns `[f(x)]`.
- The internal FastDifferentiation cache is cleared before constructing the
  symbolic derivative chain.
- This backend is useful when symbolic derivative reuse is preferable to
  repeated AD calls.
"""
function derivative_jet_fastdifferentiation(
    f,
    x::Real,
    nmax::Int
)
    nmax >= 0 || throw(ArgumentError("nmax must be ≥ 0 (got nmax=$nmax)"))

    x0 = float(x)

    if nmax == 0
        return [float(f(x0))]
    end

    FastDifferentiation.clear_cache()

    @variables t

    base_expr = f(t)

    exprs = Vector{Any}(undef, nmax + 1)
    exprs[1] = base_expr

    for n in 1:nmax
        exprs[n + 1] = FastDifferentiation.derivative(
            base_expr,
            ntuple(_ -> t, n)...
        )
    end

    exe = FastDifferentiation.make_function(exprs, [t])
    vals = exe(x0)

    return Float64.(vec(vals))
end

end  # module ADFastDifferentiation


