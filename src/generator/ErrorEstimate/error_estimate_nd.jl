# ============================================================================
# src/generator/ErrorEstimate/error_estimate_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_nd(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        dim::Int,
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> Float64

Estimate an axis-separable tensor-product truncation-error *model* for an arbitrary-dimensional
composite Newton-Cotes rule on the hypercube ``[a,b]^\\texttt{dim}`` using the exact midpoint residual expansion.

# Function description
This routine provides a dimension-generic version of the axis-separated midpoint residual model.

Let ``\\displaystyle{h = \\frac{b-a}{N}}`` and the physical midpoint along any axis be
``\\displaystyle{\\bar{x} = \\frac{a+b}{2}}``.
From the exact rational composite weight assembly for `(rule, boundary, N)`, the midpoint residual
expansion yields a sequence of nonzero residual orders ``k`` with exact coefficients
``\\displaystyle{\\texttt{coeff}_k = \\frac{\\texttt{diff}_k}{k!}}``.

This routine collects the first `nerr_terms` nonzero residual orders
``k_1, k_2, \\ldots`` (up to `kmax`) and returns the summed axis-separable model:
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i}\\, h^{k_i+1}\\,
\\sum_{\\mu=1}^{\\texttt{dim}} I_{\\mu}^{(k_i)} \\,,
```
where, for each axis ``\\mu``,

```math
I_{\\mu}^{(k)} =
\\int \\cdots \\int
\\left( \\prod_{\\nu \\neq \\mu} dx_{\\nu} \\right)
\\; \\frac{\\partial^k f}{\\partial x_{\\mu}^k}
\\left( x_1, \\ldots, x_{\\mu}=\\bar{x}, \\ldots, x_{\\texttt{dim}} \\right) \\,.
```
Numerically, each cross-axis integral is computed by enumerating the ``(\\texttt{dim}-1)``-fold
tensor-product grid over the ``1``-dimensional nodes `xs`, accumulating the product weights, and evaluating
[`nth_derivative`](@ref) on the resulting ``1``-dimensional slice (with the selected coordinate left as the differentiation variable).

Special case:

* If `nerr_terms == 1`, this reduces to the usual leading-order (LO) axis-separable term.

# Implementation notes

* The helper `_call_with_axis` constructs the argument tuple for `f` by replacing only one coordinate (`axis`)
  with the differentiation variable `x` (which may be a Dual).
* The enumeration over the `(\\texttt{dim}-1)` indices is implemented in odometer style.
* For `dim == 1`, the routine falls back to a direct derivative evaluation.

# Arguments

* `f`:
  Callable integrand expecting exactly `dim` positional arguments.
* `a`, `b`:
  Scalar bounds defining the hypercube ``[a,b]^\\texttt{dim}``.
* `N`:
  Number of subintervals per axis. Must satisfy the composite tiling constraint for `(rule, boundary)`.
* `rule`:
  Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
* `boundary`:
  Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
* `dim`:
  Dimensionality of the integral (must satisfy ``\\texttt{dim} \\ge 1``).

# Keyword arguments

* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model (`1` = LO only, `2` = LO+NLO, ...).
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `Float64`:
  The summed axis-separable truncation-error model value.

# Errors

* Throws `ArgumentError` if `dim < 1`.
* Propagates errors from:

  * composite weight assembly / midpoint residual extraction,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`Maranatha.JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
  insufficient nonzero residual terms exist up to `kmax`.

# Notes

* The model is *axis-separated* (sum of single-axis error operators).
  Mixed-derivative contributions are higher order and intentionally omitted.
* Complexity grows as ``\\mathcal{O}\\left( n_{\\text{err}} \\; \\texttt{dim} \\; (N+1)^{\\texttt{dim}-1} \\right)`` derivative evaluations,
  so this estimator can be expensive for large `dim` at high resolution.
"""
function error_estimate_nd(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int,
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2

    xs, ws = quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    # helper: call f with axis value replaced by x (x may be Dual)
    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    # Collect residual terms (LO or LO+NLO+...)
    ks = Int[]
    coeffsR = Quadrature.RBig[]

    if nerr_terms == 1
        k, coeffR = _leading_midpoint_residual_term(
            rule, boundary, N; 
            kmax = kmax
        )
        k == 0 && return 0.0
        push!(ks, k)
        push!(coeffsR, coeffR)
    else
        ks, coeffsR = _leading_midpoint_residual_terms(
            rule, boundary, N;
            nterms = nerr_terms,
            kmax   = kmax
        )
        isempty(ks) && return 0.0
    end

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, dim - 1)

    err_total = 0.0

    @inbounds for it in eachindex(ks)
        k = ks[it]
        k == 0 && continue

        coeff = Float64(coeffsR[it])
        total_axes = 0.0

        for axis in 1:dim
            Iaxis = 0.0

            if dim == 1
                Iaxis = nth_derivative(
                    x -> f(x),
                    x̄, k;
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
                        x̄, k;
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

        err_total += coeff * h^(k+1) * total_axes
    end

    return err_total
end