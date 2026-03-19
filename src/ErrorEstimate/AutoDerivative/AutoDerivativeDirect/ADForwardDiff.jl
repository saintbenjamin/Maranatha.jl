# ============================================================================
# src/ErrorEstimate/AutoDerivativeDirect/ADForwardDiff.jl
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
    nth_derivative_forwarddiff(
        f,
        x::Real,
        n::Int
    )

Compute the ``n``-th derivative of a scalar callable `f` at `x`
using repeated [`ForwardDiff.derivative`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.derivative).

# Function description
This routine constructs a nested derivative closure chain of length ``n``,
then evaluates the resulting callable at ``x``.

It is intentionally written to accept any Julia callable, not only subtypes of
`Function`, so that closures and callable structs are also supported.

# Arguments
- `f`: Scalar-to-scalar callable.
- `x::Real`: Evaluation point.
- `n::Int`: Derivative order.

# Returns
- The `n`-th derivative value ``f^{(n)}(x)``.

# Errors
- No explicit validation is performed here; any differentiation failure from
  [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/) is propagated.

# Notes
- This is the default practical backend in the current error-estimation stack.
- The callable restriction `f::Function` is intentionally avoided.
"""
function nth_derivative_forwarddiff(
    f,
    x::Real,
    n::Int
)
    T = typeof(x)
    g = f
    for _ in 1:n
        prev = g
        g = t -> ForwardDiff.derivative(prev, t)
    end
    return convert(T, g(convert(T, x)))
end

end  # module ADForwardDiff