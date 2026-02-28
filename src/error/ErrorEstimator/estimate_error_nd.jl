# ============================================================================
# src/error/ErrorEstimator/estimate_error_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_nd(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        dim::Int
    ) -> Float64

Estimate the leading tensor-product truncation error for an arbitrary-dimensional
composite Newton-Cotes rule on the hypercube ``[a,b]^\\texttt{dim}``
using the exact midpoint residual expansion.

# Function description
This routine provides a dimension-generic version of the axis-separated midpoint error model.

Let ``\\displaystyle{h = \\frac{b-a}{N}}``.
From the exact rational composite weight assembly for
`(rule, boundary, N)`, it determines:

- the leading midpoint residual order ``k``, and
- its exact rational coefficient `coeffR`.

The model returned is:
```math
E = \\texttt{coeff} \\, h^{k+1} \\, \\sum_{\\mu=1}^{\\texttt{dim}} I_\\mu
```
where `coeff = Float64(coeffR)` and each ``I_\\mu`` is the tensor-product integral
over the remaining ``\\texttt{dim}-1`` coordinates of the ``k``-th partial derivative along
the selected axis evaluated at the physical midpoint along that axis:

- midpoint along any axis: ``\\displaystyle{\\bar{x} = \\frac{a+b}{2}}``
- for each axis ``\\mu``, define:
```math
I_\\mu = \\int \\cdots \\int \\left( \\prod_{\\nu \\neq \\mu} dx_\\nu \\right) \\; \\frac{\\partial^k f}{\\partial x_\\mu^k} \\left( x_1 , \\ldots , x_\\mu=\\bar{x} , \\ldots , x_{\\texttt{dim}} \\right)
```
Numerically, the cross-axis integral is computed by enumerating the ``(\\texttt{dim}-1)``-fold
tensor-product grid over the 1D nodes `xs`, accumulating the product weights, and
evaluating `nth_derivative` on the resulting 1D slice (with one coordinate left as
the differentiation variable).

# Implementation notes
- The helper `_call_with_axis` constructs the argument tuple for `f` by replacing
  only one coordinate (`axis`) with the differentiation variable `x` (which may be a Dual).
- The enumeration over the ``(\\texttt{dim}-1)`` indices is implemented in odometer style.
- For `dim == 1`, the routine falls back to a direct derivative evaluation.

# Arguments
- `f`:
    Callable integrand expecting exactly `dim` positional arguments.
- `a`, `b`:
    Scalar bounds defining the hypercube ``[a,b]^\\texttt{dim}``.
- `N`:
    Number of subintervals per axis.
    Must satisfy the composite tiling constraint for `(rule, boundary)`.
- `rule`:
    Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
- `boundary`:
    Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
- `dim`:
    Dimensionality of the integral (must satisfy ``\\texttt{dim} \\ge 1``).

# Returns
- `Float64`:
    Leading tensor-product truncation error estimate in `dim` dimensions.

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Propagates any errors from:
  - composite weight assembly,
  - midpoint residual extraction,
  - derivative evaluation ([`nth_derivative`](@ref)).

# Notes
- The model is *axis-separated* (sum of single-axis error operators).
  Mixed-derivative contributions and higher residual orders are omitted.
- Returns `0.0` if the residual scan reports `k == 0` (degenerate/unexpected case).
- Complexity grows as ``O(\\texttt{dim} \\, (N+1)^{\\texttt{dim}-1})`` derivative evaluations, so this
  estimator can be expensive for large `dim` at high resolution.
"""
function estimate_error_nd(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xs, ws = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    # helper: call f with axis value replaced by x (x may be Dual)
    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    # ---- default tensor-style midpoint model (β-residual based) ----

    k, coeffR = _leading_midpoint_residual_term(rule, boundary, N; kmax=64)
    k == 0 && return 0.0
    coeff = Float64(coeffR)

    xmid = (aa + bb) / 2

    total_axes = 0.0

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, dim - 1)

    @inbounds for axis in 1:dim
        Iaxis = 0.0

        if dim == 1
            Iaxis = nth_derivative(
                x -> f(x),
                xmid, k;
                h=h, rule=rule, N=N, dim=dim,
                side=:mid, axis=axis, stage=:midpoint
            )
        else
            fill!(idx, 1)
            while true
                wprod = 1.0
                t = 1
                for d in 1:dim
                    if d == axis
                        continue
                    end
                    i = idx[t]
                    fixed[d] = xs[i]
                    wprod *= ws[i]
                    t += 1
                end

                Iaxis += wprod * nth_derivative(
                    x -> _call_with_axis(f, fixed, axis, x, dim),
                    xmid, k;
                    h=h, rule=rule, N=N, dim=dim,
                    side=:mid, axis=axis, stage=:midpoint
                )

                # odometer increment
                q = dim - 1
                while q >= 1
                    idx[q] += 1
                    if idx[q] <= length(xs)
                        break
                    else
                        idx[q] = 1
                        q -= 1
                    end
                end
                q == 0 && break
            end
        end

        total_axes += Iaxis
    end

    return coeff * h^(k+1) * total_axes
end