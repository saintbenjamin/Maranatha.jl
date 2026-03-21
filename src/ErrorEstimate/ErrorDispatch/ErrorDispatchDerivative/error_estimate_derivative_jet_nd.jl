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

Estimate an arbitrary-dimensional axis-separable midpoint-residual
truncation-error model using derivative-jet reuse.

# Function description
This routine builds the same generic ``n``-dimensional asymptotic
midpoint-residual error model as [`error_estimate_derivative_direct_nd`](@ref),
but instead of requesting each derivative order independently, it evaluates the
required derivatives through shared jet-based calls along each axis slice.

Two domain conventions are supported:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as
  ``[a,b]^{\\texttt{dim}}``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `dim`, the domain is interpreted as
  ``[a_1,b_1] \\times \\cdots \\times [a_{\\texttt{dim}}, b_{\\texttt{dim}}]``.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`AutoDerivativeJet._derivative_values_for_ks`](@ref)
to each slice callable in order to reuse one derivative jet per slice rather
than one scalar derivative call per requested order.

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
  jet-based derivative evaluation.

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
- Unlike the 2D/3D/4D compatibility wrappers, this generic `nd` variant returns
  the exact axis-wise decomposition in `per_axis` rather than a flattened
  legacy summary.
"""
function error_estimate_derivative_jet_nd(
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

    xs_list = Vector{Vector{T}}(undef, dim)
    ws_list = Vector{Vector{T}}(undef, dim)

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
        xs_list[d] = collect(T, xd)
        ws_list[d] = collect(T, wd)
    end

    jet_fun, backend_tag =
        AutoDerivativeJet.resolve_derivative_jet_backend(err_method)

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

    @inline function call_axis(f, fixed, axis, x, dim)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    axis_results = Vector{NamedTuple}(undef, dim)

    for axis in 1:dim
        ks = ks_axes[axis]
        coeffs = coeffs_axes[axis]

        isempty(ks) && begin
            axis_results[axis] = (;
                ks = Int[],
                coeffs = T[],
                derivatives = T[],
                terms = T[],
                total = zero(T),
            )
            continue
        end

        derivatives = zeros(T, length(ks))
        terms       = zeros(T, length(ks))

        if dim == 1
            vals0 = AutoDerivativeJet._derivative_values_for_ks(
                jet_fun,
                backend_tag,
                x -> f(x),
                center[1],
                ks;
            )
            vals = T.(vals0)

            for it in eachindex(ks)
                k = ks[it]
                k == 0 && continue
                derivatives[it] += vals[it]
            end
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
                    fixed[d] = xs_list[d][i]
                    wprod *= ws_list[d][i]
                    t += 1
                end

                vals0 = AutoDerivativeJet._derivative_values_for_ks(
                    jet_fun,
                    backend_tag,
                    x -> call_axis(f, fixed, axis, x, dim),
                    center[axis],
                    ks;
                )
                vals = T.(vals0)

                for it in eachindex(ks)
                    k = ks[it]
                    k == 0 && continue
                    derivatives[it] += wprod * vals[it]
                end

                q = dim - 1
                while q >= 1
                    idx[q] += 1
                    other_axis = q >= axis ? q + 1 : q

                    if idx[q] <= length(xs_list[other_axis])
                        break
                    else
                        idx[q] = 1
                        q -= 1
                    end
                end
                q == 0 && break
            end
        end

        for it in eachindex(ks)
            k = ks[it]
            if k == 0
                derivatives[it] = zero(T)
                terms[it] = zero(T)
            else
                terms[it] = coeffs[it] * hs[axis]^(k + 1) * derivatives[it]
            end
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
