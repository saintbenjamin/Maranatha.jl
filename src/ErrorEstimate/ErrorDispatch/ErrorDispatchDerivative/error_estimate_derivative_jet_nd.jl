# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet_nd(
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

Estimate an arbitrary-dimensional axis-separable midpoint-residual
truncation-error model using derivative-jet reuse.

# Function description
This routine builds the same generic ``n``-dimensional asymptotic
midpoint-residual error model as [`error_estimate_derivative_direct_nd`](@ref), but instead of
requesting each derivative order independently, it evaluates the required
derivatives through shared jet-based calls along each axis slice.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`AutoDerivativeJet._derivative_values_for_ks`](@ref) to each
slice callable in order to reuse one derivative jet per slice rather than one
scalar derivative call per requested order.

For each collected residual order ``k``, it sums the axis-wise contributions
obtained by inserting the midpoint along one differentiation axis and
integrating over the remaining ``\\texttt{dim} - 1`` axes through an
odometer-style tensor-product traversal.

# Arguments
- `f`: Callable integrand accepting exactly `dim` scalar arguments.
- `a::Real`: Lower bound.
- `b::Real`: Upper bound.
- `N::Int`: Number of subintervals per axis.
- `rule::Symbol`: Quadrature rule symbol.
- `boundary::Symbol`: Boundary pattern symbol.

# Keyword arguments
- `dim::Int`: Problem dimensionality.
- `err_method::Symbol`: Derivative backend selector.
- `nerr_terms::Int`: Number of nonzero residual terms to include.
- `kmax::Int`: Maximum residual order scanned.

# Returns
- `NamedTuple` with fields:
  - `ks`
  - `coeffs`
  - `derivatives`
  - `terms`
  - `total`
  - `center`
  - `h`

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1`.
- Propagates quadrature-node construction, residual-model extraction, and
  jet-based derivative-evaluation errors.

# Notes
- The model is axis-separable.
- Mixed derivative terms are intentionally omitted.
- This estimator can become expensive for large `dim`.
- If no residual orders are collected, the function returns a zero-total result
  with empty arrays.
- This variant is especially useful when several residual orders are needed for
  each slice, since one shared jet can supply all requested derivatives on that
  slice.
"""
function error_estimate_derivative_jet_nd(
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

    # ks, coeffs, _center = _leading_residual_terms_any(
    #     rule, boundary, N;
    #     nterms = nerr_terms,
    #     kmax   = kmax
    # )
    ks, coeffs, _center = _get_residual_model_fixed(
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

    derivatives = zeros(Float64, length(ks))
    terms       = zeros(Float64, length(ks))

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, max(dim - 1, 1))

    if dim == 1
        vals = AutoDerivativeJet._derivative_values_for_ks(
            x -> f(x),
            x̄,
            ks;
            h = h,
            rule = rule,
            N = N,
            dim = dim,
            err_method = err_method,
            side = :mid,
            axis = 1,
            stage = :midpoint,
        )

        @inbounds for it in eachindex(ks)
            k = ks[it]
            if k == 0
                derivatives[it] = 0.0
                terms[it] = 0.0
            else
                derivatives[it] = vals[it]
                terms[it] = coeffs[it] * h^(k + 1) * derivatives[it]
            end
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

    for axis in 1:dim
        fill!(idx, 1)

        while true
            wprod = 1.0
            t = 1

            @inbounds for d in 1:dim
                if d == axis
                    continue
                end
                i = idx[t]
                fixed[d] = xs[i]
                wprod *= ws[i]
                t += 1
            end

            vals = AutoDerivativeJet._derivative_values_for_ks(
                x -> _call_with_axis(f, fixed, axis, x, dim),
                x̄,
                ks;
                h = h,
                rule = rule,
                N = N,
                dim = dim,
                err_method = err_method,
                side = :mid,
                axis = axis,
                stage = :midpoint,
            )

            @inbounds for it in eachindex(ks)
                k = ks[it]
                k == 0 && continue
                derivatives[it] += wprod * vals[it]
            end

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

    @inbounds for it in eachindex(ks)
        k = ks[it]
        if k == 0
            derivatives[it] = 0.0
            terms[it] = 0.0
        else
            terms[it] = coeffs[it] * h^(k + 1) * derivatives[it]
        end
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