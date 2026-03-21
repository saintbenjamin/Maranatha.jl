# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/error_estimate_derivative_jet_3d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    error_estimate_derivative_jet_3d(
        f,
        a,
        b,
        N::Int,
        rule,
        boundary;
        err_method::Symbol = :forwarddiff,
        nerr_terms::Int = 1,
        kmax::Int = 128,
        real_type = nothing,
    )

Estimate a ``3``-dimensional axis-separable midpoint-residual truncation-error
model using derivative-jet reuse.

# Function description
This routine builds the same ``3``-dimensional asymptotic midpoint-residual
error model as [`error_estimate_derivative_direct_3d`](@ref), but instead of
requesting each derivative order independently, it evaluates the required
derivatives through shared jet-based calls along each axis slice.

Two domain conventions are supported:

- **Hypercube-style input**:
  if `a` and `b` are scalar bounds, the domain is interpreted as ``[a,b]^3``.

- **Axis-wise rectangular input**:
  if `a` and `b` are tuples or vectors of length `3`, the domain is interpreted as
  ``[a_1,b_1] \\times [a_2,b_2] \\times [a_3,b_3]``.

The residual-term model is obtained through the cached helper
[`_get_residual_model_fixed`](@ref). After the residual orders ``k_i`` are
identified, the function applies [`AutoDerivativeJet._derivative_values_for_ks`](@ref)
to each slice function in order to reuse one derivative jet per slice rather
than one scalar derivative call per requested order.

For each collected residual order ``k``, it forms the model contribution
```math
E \\approx \\sum_{i=1}^{n_{\\text{err}}}
\\texttt{coeff}_{k_i} \\, h^{k_i+1} \\, \\left( I_x^{(k_i)} + I_y^{(k_i)} + I_z^{(k_i)} \\right) \\, .
```

# Arguments

* `f`: Scalar callable integrand `f(x, y, z)`.
* `a`:
  Lower integration bound specification.
  This may be either a scalar lower bound shared across all axes, or a length-3
  tuple/vector of per-axis lower bounds.
* `b`:
  Upper integration bound specification.
  This may be either a scalar upper bound shared across all axes, or a length-3
  tuple/vector of per-axis upper bounds.
* `N::Int`: Number of subintervals per axis.
* `rule`: Quadrature rule specification.
  This may be either a scalar rule symbol shared across all three axes, or a
  length-3 tuple/vector of per-axis rule symbols.
* `boundary`: Boundary pattern specification.
  This may be either a scalar boundary symbol shared across all three axes, or a
  length-3 tuple/vector of per-axis boundary symbols.

# Keyword arguments

* `err_method::Symbol`: Derivative backend selector.
* `nerr_terms::Int`: Number of nonzero residual terms to include.
* `kmax::Int`: Maximum residual order scanned.
* `real_type = nothing`:
  Optional scalar type used internally for bound conversion, quadrature nodes
  and weights, residual-coefficient conversion, and jet-based derivative
  evaluation.

# Returns

* `NamedTuple` with fields:

  * `ks`
  * `coeffs`
  * `derivatives`
  * `terms`
  * `total`
  * `center`
  * `h`
  * `per_axis`

# Errors

* Throws `ArgumentError` if axis-wise bounds are supplied but `length(a) != 3`
  or `length(b) != 3`.
* Throws (via [`JobLoggerTools.error_benji`](@ref)) if `nerr_terms < 1` or `kmax < 0`.
* Propagates quadrature-node construction, residual-model extraction, and
  jet-based derivative-evaluation errors.

# Notes

* Only axis-separable contributions are included.
* Mixed derivative terms are intentionally omitted.
* This variant is especially useful when several residual orders are needed for
  each slice, since one shared jet can supply all requested derivatives on that
  slice.
"""
function error_estimate_derivative_jet_3d(
    f,
    a,
    b,
    N::Int,
    rule,
    boundary;
    err_method::Symbol = :forwarddiff,
    nerr_terms::Int = 1,
    kmax::Int = 128,
    real_type = nothing,
)
    return _flatten_axiswise_error_result(
        error_estimate_derivative_jet_nd(
            f,
            a,
            b,
            N,
            rule,
            boundary;
            dim = 3,
            err_method = err_method,
            nerr_terms = nerr_terms,
            kmax = kmax,
            real_type = real_type,
        )
    )
end
