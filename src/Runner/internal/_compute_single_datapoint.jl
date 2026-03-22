# ============================================================================
# src/Runner/internal/_compute_single_datapoint.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _compute_single_datapoint(
        integrand,
        aT,
        bT,
        N::Int,
        dim::Int,
        rule,
        boundary;
        is_rect_domain::Bool,
        err_method::Symbol,
        nerr_terms::Int,
        use_error_jet::Bool,
        threaded_subgrid::Bool,
        use_cuda::Bool,
        T,
    ) -> NamedTuple

Compute one quadrature/error-estimation datapoint for the runner.

# Function description
This helper performs the per-resolution work inside the main runner loop. For a
single subdivision count `N`, it constructs the step object `h`, derives the
scalar step proxy `h_l2`, evaluates the quadrature estimate, and evaluates the
corresponding error-scale model.

The quadrature estimate is forwarded as `I_coarse` to the error-estimation
dispatcher so the error estimator can reuse the already computed coarse value
when applicable.

# Arguments
- `integrand`:
  Integrand callable used for quadrature and error estimation.
- `aT`, `bT`:
  Domain bounds already normalized to the active scalar type.
- `N::Int`:
  Subdivision count for the current datapoint.
- `dim::Int`:
  Problem dimension.
- `rule`:
  Quadrature-rule specification.
- `boundary`:
  Boundary specification.
- `is_rect_domain::Bool`:
  Whether the domain is axis-wise rectangular.
- `err_method::Symbol`:
  Active error-estimation mode.
- `nerr_terms::Int`:
  Number of derivative-residual terms requested by the error model.
- `use_error_jet::Bool`:
  Whether the derivative-based branch should use the jet path.
- `threaded_subgrid::Bool`:
  Whether threaded-subgrid quadrature should be enabled.
- `use_cuda::Bool`:
  Whether CUDA execution should be requested.
- `T`:
  Active scalar type for the current run.

# Returns
- `NamedTuple` with fields:
  - `h`: original step object for the current resolution,
  - `h_l2`: scalar step proxy stored in the runner result,
  - `estimate`: quadrature estimate,
  - `err`: error-estimator output object.

# Errors
- Propagates quadrature errors from [`QuadratureDispatch.quadrature`](@ref).
- Propagates error-estimation errors from [`ErrorDispatch.error_estimate`](@ref).
- Propagates arithmetic or conversion errors arising during step construction.

# Notes
- For scalar domains, `h_l2` is the scalar step size itself.
- For axis-wise domains, `h_l2` is the Euclidean norm of the per-axis
  step object.
"""
function _compute_single_datapoint(
    integrand,
    aT,
    bT,
    N::Int,
    dim::Int,
    rule,
    boundary;
    is_rect_domain::Bool,
    err_method::Symbol,
    nerr_terms::Int,
    use_error_jet::Bool,
    threaded_subgrid::Bool,
    use_cuda::Bool,
    T,
)
    h = if !is_rect_domain
        (bT - aT) / T(N)
    else
        aT isa Tuple ?
            ntuple(i -> (bT[i] - aT[i]) / T(N), dim) :
            T[(bT[i] - aT[i]) / T(N) for i in 1:dim]
    end

    h_l2 = if !is_rect_domain
        h
    else
        sqrt(sum(x -> x*x, h))
    end

    I = QuadratureDispatch.quadrature(
        integrand,
        aT,
        bT,
        N,
        dim,
        rule,
        boundary;
        use_cuda = use_cuda,
        threaded_subgrid = threaded_subgrid,
        real_type = T,
    )

    err = ErrorDispatch.error_estimate(
        integrand,
        aT,
        bT,
        N,
        dim,
        rule,
        boundary;
        err_method = err_method,
        nerr_terms = nerr_terms,
        use_error_jet = use_error_jet,
        threaded_subgrid = threaded_subgrid,
        real_type = T,
        I_coarse = I,
    )

    return (;
        h = h,
        h_l2 = h_l2,
        estimate = I,
        err = err,
    )
end