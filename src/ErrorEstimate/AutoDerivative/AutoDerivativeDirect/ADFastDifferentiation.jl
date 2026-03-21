# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect/ADFastDifferentiation.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ADFastDifferentiation

FastDifferentiation-based backend for scalar direct derivative evaluation.

# Module description
This module implements the direct `n`-th-derivative backend based on
`FastDifferentiation.jl` for the automatic-differentiation layer used by
`Maranatha.ErrorEstimate`.

It differentiates symbolic expressions generated from the supplied scalar
callable and evaluates the compiled derivative expression at the target point.

# Notes
- This is an internal backend module.
- Backend selection is handled by `AutoDerivativeDirect`.
"""
module ADFastDifferentiation

import FastDifferentiation
import FastDifferentiation: @variables

"""
    nth_derivative_fastdifferentiation(
        f,
        x::Real,
        n::Int
    )

Compute the ``n``-th derivative of a scalar callable `f` at `x`
using symbolic differentiation via [`FastDifferentiation.jl`](https://brianguenter.github.io/FastDifferentiation.jl/stable/).

# Function description
This routine evaluates `f` on a symbolic variable, constructs the symbolic
`n`-th derivative expression, compiles that expression to an executable
function, and evaluates it at `x`.

# Arguments
- `f`: Scalar-to-scalar callable that must accept symbolic `Node` inputs.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

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
    T = typeof(x)
    n >= 0 || throw(ArgumentError("n must be ≥ 0 (got n=$n)"))
    n == 0 && return convert(T, f(x))

    FastDifferentiation.clear_cache()

    @variables t

    expr = f(t)
    dexpr = FastDifferentiation.derivative(expr, ntuple(_ -> t, n)...)
    exe = FastDifferentiation.make_function([dexpr], [t])

    return convert(T, exe(convert(T, x))[1])
end

end  # module ADFastDifferentiation
