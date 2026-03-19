# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_direct_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_direct_nd(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol,
        boundary::Symbol;
        dim::Int,
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128,
        real_type = nothing,
    )

Estimate an arbitrary-dimensional axis-separable midpoint-residual truncation-error model.

# Function description
This routine provides the generic ``n``-dimensional version of the
midpoint-residual model on the hypercube ``[a,b]^n``.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref), which reuses previously constructed
residual data for the same rule configuration when available.

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
- `real_type = nothing`:
  Optional scalar type used internally for bound conversion, quadrature nodes
  and weights, residual-coefficient conversion, midpoint placement, and
  derivative evaluation.

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
  derivative-evaluation errors.

# Notes
- The model is axis-separable.
- Mixed derivative terms are intentionally omitted.
- This estimator can become expensive for large `dim`.
- Residual-term reuse through caching reduces repeated setup cost across
  multiple calls with the same rule configuration.
"""
function error_estimate_derivative_direct_nd(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol,
    boundary::Symbol;
    dim::Int,
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")

    aa = convert(T, a)
    bb = convert(T, b)
    h  = (bb - aa) / T(N)

    x̄ = (aa + bb) / T(2)

    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(
        aa, bb, N, rule, boundary;
        real_type = T,
    )

    @inline function _call_with_axis(f, fixed, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    ks, coeffs0, _center = _get_residual_model_fixed(
        rule, boundary, N;
        nterms = nerr_terms,
        kmax   = kmax
    )
    coeffs = T.(coeffs0)

    isempty(ks) && return (;
        ks = Int[],
        coeffs = T[],
        derivatives = T[],
        terms = T[],
        total = zero(T),
        center = ntuple(_ -> x̄, dim),
        h = h
    )

    derivatives = Vector{T}(undef, length(ks))
    terms       = Vector{T}(undef, length(ks))

    fixed = Vector{T}(undef, dim)
    idx   = ones(Int, dim - 1)

    deriv_fun, backend_tag = AutoDerivativeDirect.resolve_nth_derivative_backend(err_method)

    @inbounds for it in eachindex(ks)
        k = ks[it]

        if k == 0
            derivatives[it] = zero(T)
            terms[it] = zero(T)
            continue
        end

        coeff = coeffs[it]
        total_axes = zero(T)

        for axis in 1:dim
            Iaxis = zero(T)

            if dim == 1
                Iaxis = convert(T, AutoDerivativeDirect.nth_derivative(
                    deriv_fun,
                    backend_tag,
                    x -> f(x),
                    x̄, 
                    k;
                ))
            else
                fill!(idx, 1)

                while true
                    wprod = one(T)
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

                    Iaxis += wprod * convert(T, AutoDerivativeDirect.nth_derivative(
                        deriv_fun,
                        backend_tag,
                        x -> _call_with_axis(f, fixed, axis, x, dim),
                        x̄, 
                        k;
                    ))

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