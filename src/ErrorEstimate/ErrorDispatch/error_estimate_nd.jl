# ============================================================================
# src/ErrorEstimate/ErrorDispatch/error_estimate_nd.jl
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
        err_method::Symbol = :forwarddiff,
        dim::Int,
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> NamedTuple

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

The routine returns the full decomposition of the asymptotic error model,
including individual residual contributions and their summed value.

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
  Composite Newton-Cotes rule symbol (must be `:newton_pK` style).
* `boundary`:
  Boundary pattern (`:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`).
* `dim`:
  Dimensionality of the integral (must satisfy ``\\texttt{dim} \\ge 1``).

# Keyword arguments

* `err_method`:
  Backend used for derivative evaluation via [`nth_derivative`](@ref).
  Supported values: `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, `:enzyme`.
* `nerr_terms`:
  Number of nonzero midpoint residual terms to include in the model (`1` = LO only, `2` = LO+NLO, ...).
* `kmax`:
  Maximum residual order scanned when collecting terms.

# Returns

* `NamedTuple` with fields:

  * `ks` - residual orders used in the model
  * `coeffs` - midpoint residual coefficients
  * `derivatives` - evaluated derivatives ``f^{(k)}(\\bar{x})``
  * `terms` - individual asymptotic error contributions
  * `total` - summed truncation-error model value
  * `center` - midpoint ``\\bar{x}``
  * `h` - step size

# Errors

* Throws `ArgumentError` if `dim < 1`.
* Propagates errors from:

  * composite weight assembly / midpoint residual extraction,
  * derivative evaluation ([`nth_derivative`](@ref)).
* Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or if
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
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    x̄ = (aa + bb) / 2

    xs, ws = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    isempty(ks) && return (;
        ks = Int[],
        coeffs = Float64[],
        derivatives = Float64[],
        terms = Float64[],
        total = 0.0,
        center = ntuple(_ -> x̄, dim),
        h = h
    )

    derivatives = Vector{Float64}(undef, length(ks))
    terms       = Vector{Float64}(undef, length(ks))

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, dim - 1)

    @inbounds for it in eachindex(ks)
        k = ks[it]

        if k == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]
        total_axes = 0.0

        for axis in 1:dim
            Iaxis = 0.0

            if dim == 1
                Iaxis = nth_derivative(
                    x -> f(x),
                    x̄, k;
                    h=h, rule=rule, N=N, dim=dim,
                    side=:mid, axis=axis, stage=:midpoint,
                    err_method=err_method
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
                        side=:mid, axis=axis, stage=:midpoint,
                        err_method=err_method
                    )

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

        derivatives[it] = total_axes
        terms[it] = coeff * h^(k + 1) * total_axes
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = ntuple(_ -> x̄, dim),
        h           = h
    )
end

"""
    error_estimate_nd_threads(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        err_method::Symbol = :forwarddiff,
        dim::Int,
        nerr_terms::Int = 1,
        kmax::Int = 128
    ) -> NamedTuple

Threaded variant of [`error_estimate_nd`](@ref) for nD midpoint-residual truncation-error modeling.

All non-threading details (mathematical definition, coefficient construction, residual-term
interpretation, and overall intent) are identical to [`error_estimate_nd`](@ref).
See that function for the full formalism and background.

# Threading implementation

This function parallelizes the axis-wise accumulation using Julia's built-in multithreading:

* For each residual order `k` in `ks`, the total contribution is a sum over axes `axis = 1:dim`.
* The per-axis contributions are distributed via [`Base.Threads.@threads`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads) over `axis in 1:dim`.
* Each axis worker performs the full `(dim-1)`-dimensional node-product summation (odometer loop)
  for its assigned axis, repeatedly evaluating the required `k`-th derivative at the midpoint.
* Per-axis results are accumulated into a thread-local `Float64` buffer indexed by [`Threads.threadid()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.threadid),
  followed by a `sum` reduction across threads.
* Thread safety is ensured by allocating `fixed` and `idx` buffers *inside* the threaded loop,
  avoiding any shared mutable state across threads.

Threading is enabled when Julia is started with `JULIA_NUM_THREADS > 1`.

# Arguments

Same as [`error_estimate_nd`](@ref), with `dim` selecting the dimensionality.

# Keyword arguments

Same as [`error_estimate_nd`](@ref).

# Returns

Same as [`error_estimate_nd`](@ref).

# Notes

* This is an asymptotic *model* (fit stabilization / scaling diagnostics), not a strict bound.
* For small `dim` or small `nerr_terms`, threading overhead may dominate.
"""
function error_estimate_nd_threads(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int,
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N
    x̄ = (aa + bb) / 2

    xs, ws = QuadratureDispatch.get_quadrature_1d_nodes_weights(aa, bb, N, rule, boundary)

    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    ks, coeffs, _center = _leading_residual_terms_any(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )

    isempty(ks) && return (;
        ks = Int[],
        coeffs = Float64[],
        derivatives = Float64[],
        terms = Float64[],
        total = 0.0,
        center = ntuple(_ -> x̄, dim),
        h = h
    )

    nt = Threads.nthreads()

    derivatives = Vector{Float64}(undef, length(ks))
    terms       = Vector{Float64}(undef, length(ks))

    @inbounds for it in eachindex(ks)
        k = ks[it]

        if k == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
            continue
        end

        coeff = coeffs[it]

        axis_parts = zeros(Float64, nt)

        Threads.@threads for axis in 1:dim
            tid = Threads.threadid()

            fixed = Vector{Float64}(undef, dim)
            idx   = ones(Int, dim - 1)

            Iaxis = 0.0

            if dim == 1
                Iaxis = nth_derivative(
                    x -> f(x),
                    x̄, k;
                    h=h, rule=rule, N=N, dim=dim,
                    side=:mid, axis=axis, stage=:midpoint,
                    err_method=err_method
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
                        side=:mid, axis=axis, stage=:midpoint,
                        err_method=err_method
                    )

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

            axis_parts[tid] += Iaxis
        end

        total_axes = sum(axis_parts)

        derivatives[it] = total_axes
        terms[it] = coeff * h^(k + 1) * total_axes
    end

    return (;
        ks          = ks,
        coeffs      = coeffs,
        derivatives = derivatives,
        terms       = terms,
        total       = sum(terms),
        center      = ntuple(_ -> x̄, dim),
        h           = h
    )
end