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
        a,
        b,
        N::Int,
        rule,
        boundary;
        dim::Int,
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128,
        real_type = nothing,
    )

Estimate an arbitrary-dimensional axis-separable midpoint-residual truncation-error model.

# Function description
This routine provides the generic ``n``-dimensional version of the
midpoint-residual model.

Two domain conventions are supported:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, the domain is interpreted as
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref), which reuses previously constructed
residual data for the same rule configuration when available.

For each collected residual order ``k``, it sums the axis-wise contributions
obtained by inserting the midpoint along one differentiation axis and
integrating over the remaining ``\\texttt{dim} - 1`` axes through an
odometer-style tensor-product traversal.

# Arguments
- `f`: Callable integrand accepting exactly `dim` scalar arguments.
- `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a tuple/vector
  of per-axis lower bounds of length `dim`.
- `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a tuple/vector
  of per-axis upper bounds of length `dim`.
- `N::Int`: Number of subintervals per axis.
- `rule`: Quadrature rule specification.
  This may be either a scalar rule symbol shared across all axes, or a
  tuple/vector of per-axis rule symbols of length `dim`.
- `boundary`: Boundary pattern specification.
  This may be either a scalar boundary symbol shared across all axes, or a
  tuple/vector of per-axis boundary symbols of length `dim`.

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
  - `per_axis`
  - `total`
  - `center`
  - `h`

# Errors
- Throws `ArgumentError` if `dim < 1`.
- Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != dim`
  or `length(b) != dim`.
- Throws `ArgumentError` if an axis-wise `rule` or `boundary` specification has
  length different from `dim`.
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1`.
- Propagates quadrature-node construction, residual-model extraction, and
  derivative-evaluation errors.

# Notes
- The model is axis-separable.
- Mixed derivative terms are intentionally omitted.
- This estimator can become expensive for large `dim`.
- Residual-term reuse through caching reduces repeated setup cost across
  multiple calls with the same rule configuration.
- Unlike the 2D/3D/4D compatibility wrappers, this generic `nd` variant returns
  the exact axis-wise decomposition in `per_axis` rather than a flattened
  legacy summary.
"""
function error_estimate_derivative_direct_nd(
    f,
    a,
    b,
    N::Int,
    rule,
    boundary;
    dim::Int,
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128,
    real_type = nothing,
)
    T = isnothing(real_type) ? promote_type(typeof(a), typeof(b)) : real_type

    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    (nerr_terms >= 1) || JobLoggerTools.error_benji("nerr_terms must be ≥ 1")
    (kmax >= 0)       || JobLoggerTools.error_benji("kmax must be ≥ 0")

    if !(a isa AbstractVector || a isa Tuple)
        aa = ntuple(_ -> convert(T, a), dim)
        bb = ntuple(_ -> convert(T, b), dim)
    else
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))

        aa = ntuple(i -> convert(T, a[i]), dim)
        bb = ntuple(i -> convert(T, b[i]), dim)
    end

    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    hs = ntuple(i -> (bb[i] - aa[i]) / T(N), dim)
    center = ntuple(i -> (aa[i] + bb[i]) / T(2), dim)

    nodes = Vector{Vector{T}}(undef, dim)
    weights = Vector{Vector{T}}(undef, dim)

    for d in 1:dim
        xd, wd = QuadratureNodes.get_quadrature_1d_nodes_weights(
            aa[d],
            bb[d],
            N,
            rule,
            boundary;
            real_type = T,
            axis = d,
            dim = dim,
        )
        nodes[d] = collect(T, xd)
        weights[d] = collect(T, wd)
    end

    deriv_fun, backend_tag =
        AutoDerivativeDirect.resolve_nth_derivative_backend(err_method)

    ks_axes = Vector{Vector{Int}}(undef, dim)
    coeffs_axes = Vector{Vector{T}}(undef, dim)

    for d in 1:dim
        rd = QuadratureRuleSpec._rule_at(rule, d, dim)
        bd = QuadratureBoundarySpec._boundary_at(boundary, d, dim)
        ks_d, coeffs_d0, _ = _get_residual_model_fixed(
            rd,
            bd,
            N;
            nterms = nerr_terms,
            kmax = kmax,
            real_type = T,
        )
        ks_axes[d] = ks_d
        coeffs_axes[d] = T.(coeffs_d0)
    end

    fixed = Vector{T}(undef, dim)
    idx   = ones(Int, max(dim - 1, 1))

    @inline function _call_with_axis(f, fixed, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    axis_results = Vector{NamedTuple}(undef, dim)

    for axis in 1:dim
        ks = ks_axes[axis]
        coeffs = coeffs_axes[axis]

        derivatives = Vector{T}(undef, length(ks))
        terms       = Vector{T}(undef, length(ks))

        for it in eachindex(ks)
            k = ks[it]

            if k == 0
                derivatives[it] = zero(T)
                terms[it] = zero(T)
                continue
            end

            coeff = coeffs[it]
            Iaxis = zero(T)

            if dim == 1
                Iaxis = convert(
                    T,
                    AutoDerivativeDirect.nth_derivative(
                        deriv_fun,
                        backend_tag,
                        x -> f(x),
                        center[1],
                        k;
                    )
                )
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
                        fixed[d] = nodes[d][i]
                        wprod *= weights[d][i]
                        t += 1
                    end

                    Iaxis += wprod * convert(
                        T,
                        AutoDerivativeDirect.nth_derivative(
                            deriv_fun,
                            backend_tag,
                            x -> _call_with_axis(f, fixed, axis, x, dim),
                            center[axis],
                            k;
                        )
                    )

                    q = dim - 1
                    while q >= 1
                        idx[q] += 1
                        other_axis = q >= axis ? q + 1 : q
                        if idx[q] <= length(nodes[other_axis])
                            break
                        else
                            idx[q] = 1
                            q -= 1
                        end
                    end

                    q == 0 && break
                end
            end

            derivatives[it] = Iaxis
            terms[it] = coeff * hs[axis]^(k + 1) * Iaxis
        end

        axis_results[axis] = (;
            ks = ks,
            coeffs = coeffs,
            derivatives = derivatives,
            terms = terms,
            total = sum(terms),
        )
    end

    return (;
        per_axis = axis_results,
        total = sum(r.total for r in axis_results),
        center = center,
        h = hs,
    )
end
