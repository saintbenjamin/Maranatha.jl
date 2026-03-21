# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect/ADEnzyme.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module ADEnzyme

Enzyme-based backend for scalar direct derivative evaluation.

# Module description
This module implements the direct `n`-th-derivative backend based on
`Enzyme.jl` for the automatic-differentiation layer used by
`Maranatha.ErrorEstimate`.

It computes higher-order scalar derivatives by recursively applying first-order
reverse-mode differentiation to nested callables.

# Notes
- This is an internal backend module.
- Backend selection is handled by `AutoDerivativeDirect`.
"""
module ADEnzyme

import Enzyme

"""
    nth_derivative_enzyme(
        f,
        x::Real,
        n::Int
    )

Compute the ``n``-th derivative of a scalar callable `f` at `x`
using repeated [`Enzyme.gradient`](https://enzyme.mit.edu/index.fcgi/julia/stable/api/#Enzyme.gradient-Union{Tuple{N},%20Tuple{ty_0},%20Tuple{ST},%20Tuple{CS},%20Tuple{StrongZero},%20Tuple{RuntimeActivity},%20Tuple{ErrIfFuncWritten},%20Tuple{ABI},%20Tuple{ReturnPrimal},%20Tuple{F},%20Tuple{ForwardMode{ReturnPrimal,%20ABI,%20ErrIfFuncWritten,%20RuntimeActivity,%20StrongZero},%20F,%20ty_0,%20Vararg{Any,%20N}}}%20where%20{F,%20ReturnPrimal,%20ABI,%20ErrIfFuncWritten,%20RuntimeActivity,%20StrongZero,%20CS,%20ST,%20ty_0,%20N}) application.

# Function description
This routine builds a nested closure chain of length `n`. Each layer replaces
the current callable by its first derivative computed through [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/) reverse-mode
automatic differentiation. The final nested callable is then evaluated at `x`.

# Arguments
- `f`: Scalar-to-scalar callable.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Errors
- No explicit validation is performed here; backend errors from [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/) are
  propagated if the differentiation chain fails.

# Notes
- Inputs are converted to the same scalar type as `x` (not necessarily `Float64`).
- The closure chain uses repeated reverse-mode differentiation via
  [`Enzyme.jl`](https://enzyme.mit.edu/index.fcgi/julia/stable/).
- This backend is mainly useful as an experimental or benchmarking path for
  scalar higher-order differentiation.
"""
function nth_derivative_enzyme(
    f,
    x::Real,
    n::Int
)
    T = typeof(x)
    g = f
    for _ in 1:n
        prev = g
        g = t -> only(Enzyme.gradient(Enzyme.Reverse, prev, convert(T, t)))
    end
    return convert(T, g(convert(T, x)))
end

end  # module ADEnzyme
