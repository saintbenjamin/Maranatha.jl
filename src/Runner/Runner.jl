# ============================================================================
# src/Runner/Runner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Runner

High-level dataset builder for the `Maranatha.jl` convergence workflow.

`Maranatha.Runner` is responsible for constructing raw multi-resolution
quadrature datasets that can later be passed to the fitting and plotting
layers of the package.

In a standard workflow, the runner sits between the user-defined integrand
and the downstream extrapolation tools:

1. evaluate quadrature estimates at several resolutions,
2. attach derivative-based or refinement-based error-scale estimates,
3. collect the results into a uniform `NamedTuple`,
4. optionally save the dataset to disk for later reuse.

The main entry point is [`run_Maranatha`](@ref).
The module also contains private helpers for run-time type resolution,
domain normalization, option preparation, per-datapoint execution, and
optional result persistence.
For least-``\\chi^2`` extrapolation of the generated dataset, see
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
For visualization of the resulting convergence behavior, see
[`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref).
"""
module Runner

import ..DoubleFloats

import ..Utils.JobLoggerTools
import ..Utils.MaranathaIO
import ..Utils.MaranathaTOML
import ..Utils.QuadratureBoundarySpec
import ..Quadrature.QuadratureRuleSpec
import ..Quadrature.QuadratureDispatch
import ..Quadrature.NewtonCotes
import ..ErrorEstimate.ErrorDispatch

include("internal/_resolve_real_type.jl")
include("internal/_normalize_domain.jl")
include("internal/_sanitize_run_nsamples.jl")
include("internal/_prepare_run_options.jl")
include("internal/_compute_single_datapoint.jl")
include("internal/_maybe_save_result.jl")

"""
    run_Maranatha(
        integrand,
        a,
        b;
        dim::Int = 1,
        nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
        rule = :gauss_p4,
        boundary = :LU_EXEX,
        err_method::Symbol = :refinement,
        fit_terms::Int = 4,
        nerr_terms::Int = 3,
        ff_shift::Int = 0,
        use_error_jet::Bool = false,
        name_prefix::String = "Maranatha",
        save_path::Union{Nothing,AbstractString} = nothing,
        write_summary::Bool = true,
        use_cuda::Bool = false,
        real_type = nothing
    )

Run a multi-resolution quadrature study, estimate an error scale at each resolution 
using either a derivative-based or refinement-based backend, 
and return the raw convergence data needed for later ``h \\to 0`` extrapolation.


This is typically the first stage of a standard `Maranatha.jl` workflow:
use `run_Maranatha` to build a convergence dataset, then pass that dataset to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref), and optionally
visualize the result with
[`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref).

# What this function does

At the beginning of the run, this routine resolves the active scalar type,
normalizes the domain, prepares runner options, and sanitizes the requested
subdivision sequence when Newton-Cotes admissibility constraints apply.

When a derivative-based backend is selected, the runner also clears the global
derivative-based error-estimation caches via
[`ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!`](@ref).
The refinement-based path does not use these caches.

For each subdivision count `N` in `nsamples`, this routine delegates one
per-resolution datapoint computation that:

1. constructs the step size for the current subdivision count `N`,
2. evaluates the quadrature estimate via
   [`QuadratureDispatch.quadrature`](@ref),
3. evaluates the error-scale model via the unified dispatcher
   [`ErrorDispatch.error_estimate`](@ref),
4. records the resulting step size, estimate, and error information.

For scalar domains, the step size is the usual scalar quantity
``\\displaystyle{h = \\frac{b-a}{N}}``.

For axis-wise domains, the per-axis step size is constructed componentwise,
and the original step object is always recorded in `tuple_h`.
Downstream consumers that need exact per-axis step information should use
`tuple_h`.

The collected results are returned as a single `NamedTuple` and can also be
written to disk for later reuse.

# Current error-estimation path

All error estimation is delegated to the unified dispatcher
[`ErrorDispatch.error_estimate`](@ref).

The active backend is selected according to `err_method` and
`use_error_jet`:

- If `err_method == :refinement`, the refinement-based backend is used.
  In this case, `use_error_jet` is ignored.

- If `err_method != :refinement` and `use_error_jet = true`,
  a jet-based derivative estimator is used.

- If `err_method != :refinement` and `use_error_jet = false`,
  a direct derivative estimator is used.

Thus, `err_method = :refinement` explicitly selects the refinement
pipeline, while `use_error_jet` controls only the derivative-based branch.

# Arguments

* `integrand`:
  Callable integrand. It must accept `dim` scalar positional arguments.

* `a`, `b`:
  Integration bounds.

  Two domain conventions are supported:

  - **Scalar bounds**:
    if `a` and `b` are real scalars, the domain is interpreted as the
    hypercube ``[a,b]^\\mathrm{dim}``.

  - **Axis-wise bounds**:
    if `a` and `b` are tuples or vectors of length `dim`, the domain is
    interpreted as a rectangular box with per-axis bounds
    ``[a_1,b_1] \\times \\cdots \\times [a_{\\mathrm{dim}}, b_{\\mathrm{dim}}]``.

  In the axis-wise case, each coordinate axis may use a different integration interval.

# Keyword arguments

* `dim::Int = 1`:
  Tensor-product quadrature dimension.

  If `a` and `b` are tuples or vectors, their lengths must both equal `dim`.

* `nsamples = [2, 3, 4, 5, 6, 7, 8, 9]`:
  Subdivision counts used to build the convergence dataset.

* `rule = :gauss_p4`:
  Quadrature rule specification forwarded to both quadrature and error estimation.

  This may be either:

  - a single rule symbol shared by all axes, or
  - a tuple/vector of rule symbols of length `dim`.

* `boundary = :LU_EXEX`:
  Boundary specification used consistently across the quadrature pipeline.

  This may be either:

  - a single boundary symbol shared by all axes, or
  - a tuple/vector of boundary symbols of length `dim`.

* `err_method::Symbol = :refinement`:
  Error-estimation backend selector.

  - `:refinement` selects the refinement-based estimator.
  - Any other supported symbol (e.g. `:forwarddiff`, `:taylorseries`,
    ``, `:enzyme`) selects a derivative-based estimator.

  The actual dispatch is performed by
  [`ErrorDispatch.error_estimate`](@ref).

  When `err_method == :refinement` and `rule` is axis-wise, all per-axis
  rule entries must belong to the same quadrature family.

* `fit_terms::Int = 4`:
  Suggested number of basis terms for a later least-``\\chi^2`` fit.
  This value is stored in the returned result for downstream reuse.

* `nerr_terms::Int = 3`:
  Number of midpoint-residual terms used in the derivative-based error model.

* `ff_shift::Int = 0`:
  Suggested forward shift for downstream fit-power selection.
  This value is stored in the returned result but is not applied here.

* `use_error_jet::Bool = false`:
  Controls only the derivative-based error-estimation branch.

  - If `true`, a jet-based derivative estimator is used.
  - If `false`, a direct derivative estimator is used.

  This option has no effect when `err_method == :refinement`.

* `name_prefix::String = "Maranatha"`:
  Prefix used when constructing output filenames.

* `save_path::Union{Nothing,AbstractString} = nothing`:
  Output directory for optional result saving. If `nothing`, no file is written.

* `write_summary::Bool = true`:
  If `true`, write a [`TOML`](https://toml.io/en/) summary alongside the saved `JLD2` dataset.

* `use_cuda::Bool = false`:
  If `true`, evaluate quadrature and error estimation using CUDA kernels
  (currently supported only for `Float32` and `Float64` real types).

* `real_type = nothing`:
  Optional numeric scalar type used internally for all computations.
  If `nothing`, `Float64` is used. Otherwise, scalar bounds or each component
  of axis-wise bounds is converted to `real_type` before evaluation.

# Returns

A `NamedTuple` containing the raw convergence-study dataset and associated metadata.
The `err` field may therefore contain either derivative-based error objects 
or refinement-based error objects, depending on `err_method`.
Important fields include:

* `a`, `b`: integration bounds converted to the selected computation type `T`,
* `nsamples`: subdivision counts actually used to build the dataset,
* `tuple_h`: original step object stored at each resolution,
* `h`: downstream step proxy; for scalar domains this is the scalar step size,
  and for axis-wise domains this is the L2 norm of the per-axis step object,
* `avg`: quadrature estimates,
* `err`: error-estimator outputs,
* `rule`, `boundary`, `dim`, `err_method`: execution metadata,
* `nerr_terms`, `fit_terms`, `ff_shift`, `use_error_jet`,
  `use_cuda`: downstream configuration hints,
* `real_type`: string form of the computation type used for the run.

# Saving behavior

If `save_path` is provided, the result is written using
[`MaranathaIO.save_datapoint_results`](@ref).
Relative save paths are normalized against the current working directory before
directory creation and file writing.

The output filename has the form

```julia
result_\$(name_prefix)_\$(spec)_N_\$(join(sort(nsamples), "_")).jld2
```

where `spec` is:

- `\$(rule)_\$(boundary)` for scalar rule / boundary specifications, or
- `1_\$(rule_1)_\$(boundary_1)_2_\$(rule_2)_\$(boundary_2)_...` for axis-wise specifications.

If `write_summary = true`, a [`TOML`](https://toml.io/en/) summary file is written alongside it.

# Notes

* `run_Maranatha` generates datasets only; it does not perform fitting.
* The error object is an error-scale model intended for stable downstream weighting,
  not necessarily a strict truncation bound. Depending on `err_method`, it may come
  either from a derivative-based asymptotic model or from a refinement-based
  coarse-versus-fine quadrature difference.
* `fit_terms` and `ff_shift` are preserved for convenience so that later fitting code
  can reuse the same workflow settings.
* The derivative-based and refinement-based branches coexist in the public API
  and are selected through `err_method`.
  All backend selection is handled internally by
  [`ErrorDispatch.error_estimate`](@ref).
* The public runner delegates internal tasks such as type resolution, domain
  normalization, run-option preparation, per-datapoint execution, and optional
  persistence to private helpers within `Maranatha.Runner`.
* For axis-wise domains, this routine always stores the original per-axis step
  information in `tuple_h`; downstream code that needs exact axis-wise spacing
  should prefer `tuple_h` over `h`.

# Examples

Direct function call:

```julia
using Maranatha
using DoubleFloats

f(x) = sin(x)

run_result = run_Maranatha(
    f,
    Double64(0.0),
    Double64(pi);
    dim = 1,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :refinement,
    real_type = Double64,
)
```

Axis-wise rectangular domain:

```julia
using Maranatha
using DoubleFloats

run_result = run_Maranatha(
    integrand_F0000_5d,
    (
        Double64(0.0),
        Double64(0.0),
        Double64(0.0),
        Double64(0.0),
        Double64(0.0),
    ),
    (
        Double64(1.0),
        Double64(pi),
        Double64(pi),
        Double64(pi),
        Double64(pi),
    );
    dim = 5,
    nsamples = [2, 3, 4, 5, 6],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :refinement,
    real_type = Double64,
)
```

Configuration-file workflow:

```julia
using Maranatha
using DoubleFloats

run_result = run_Maranatha("./sample_1d.toml")
```

For a complete end-to-end workflow, including fitting and visualization,
see the project documentation site and the example Jupyter notebooks in
the `ipynb/` directory of this repository.
"""
function run_Maranatha(
    integrand,
    a,
    b;
    dim = 1,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method::Symbol = :refinement,
    fit_terms::Int = 4,
    nerr_terms::Int = 3,
    ff_shift::Int = 0,
    use_error_jet::Bool = false,
    name_prefix::String = "Maranatha",
    save_path::Union{Nothing,AbstractString} = nothing,
    write_summary::Bool = true,
    use_cuda::Bool = false,
    real_type = nothing,
)
    T = _resolve_real_type(real_type, use_cuda)

    domain = _normalize_domain(a, b, dim, T)
    aT = domain.aT
    bT = domain.bT
    is_rect_domain = domain.is_rect_domain

    opts = _prepare_run_options(
        nsamples,
        rule,
        boundary,
        dim,
        err_method,
        use_cuda,
    )

    nsamples = opts.nsamples
    threaded_subgrid = opts.threaded_subgrid

    estimates = T[]
    error_infos = NamedTuple[]
    hs = Vector{Any}()
    hs_l2 = Vector{T}()

    for N in nsamples
        dp = _compute_single_datapoint(
            integrand,
            aT,
            bT,
            N,
            dim,
            rule,
            boundary;
            is_rect_domain = is_rect_domain,
            err_method = err_method,
            nerr_terms = nerr_terms,
            use_error_jet = use_error_jet,
            threaded_subgrid = threaded_subgrid,
            use_cuda = use_cuda,
            T = T,
        )

        push!(hs, dp.h)
        push!(hs_l2, dp.h_l2)
        push!(estimates, dp.estimate)
        push!(error_infos, dp.err)
    end

    result = (;
        a             = aT,
        b             = bT,
        nsamples      = nsamples,
        h             = hs_l2,
        tuple_h       = hs,
        avg           = estimates,
        err           = error_infos,
        rule          = rule,
        boundary      = boundary,
        dim           = dim,
        err_method    = err_method,
        nerr_terms    = nerr_terms,
        fit_terms     = fit_terms,
        ff_shift      = ff_shift,
        use_error_jet = use_error_jet,
        use_cuda      = use_cuda,
        real_type     = string(T),
    )

    _maybe_save_result(
        result,
        save_path,
        name_prefix,
        write_summary,
    )

    return result
end

"""
    run_Maranatha(
        toml_path::AbstractString
    )

Run Maranatha from a [`TOML`](https://toml.io/en/) configuration file.

# Function description

This method provides a [`TOML`](https://toml.io/en/)-driven overload of [`run_Maranatha`](@ref).

The workflow is:

1. parse the [`TOML`](https://toml.io/en/) file via [`MaranathaTOML.parse_run_config_from_toml`](@ref)
2. validate the parsed configuration via [`MaranathaTOML.validate_run_config`](@ref)
3. load the user-defined integrand from file via [`MaranathaTOML.load_integrand_from_file`](@ref)
4. forward all recovered options to the main
   `run_Maranatha(integrand, a, b; ...)` method

This allows users to keep the integrand itself in a separate Julia source file
while storing numerical and output options in a reproducible [`TOML`](https://toml.io/en/) file.

# Arguments

`toml_path::AbstractString`
: Path to the [`TOML`](https://toml.io/en/) configuration file.

# Returns

The result object returned by the main [`run_Maranatha`](@ref)`(integrand, a, b; ...)`
execution path.

# Errors

* Propagates parsing errors from [`MaranathaTOML.parse_run_config_from_toml`](@ref).
* Propagates validation errors from [`MaranathaTOML.validate_run_config`](@ref).
* Propagates integrand-loading errors from [`MaranathaTOML.load_integrand_from_file`](@ref).
* Propagates downstream execution errors from the main runner.

# Notes

The user-defined integrand file is evaluated in an isolated module so that its
helper definitions do not pollute the main package namespace.
"""
function run_Maranatha(
    toml_path::AbstractString
)
    cfg = MaranathaTOML.parse_run_config_from_toml(toml_path)
    MaranathaTOML.validate_run_config(cfg)

    integrand = MaranathaTOML.load_integrand_from_file(
        cfg.integrand_file;
        func_name = cfg.integrand_name
    )

    T = cfg.real_type === :Float32  ? Float32  :
        cfg.real_type === :Float64  ? Float64  :
        cfg.real_type === :BigFloat ? BigFloat :
        cfg.real_type === :Double64 ? DoubleFloats.Double64 :
        error("Unsupported real_type=$(cfg.real_type)")

    return Base.invokelatest(
        run_Maranatha,
        integrand,
        cfg.a,
        cfg.b;
        dim           = cfg.dim,
        nsamples      = cfg.nsamples,
        rule          = cfg.rule,
        boundary      = cfg.boundary,
        err_method    = cfg.err_method,
        fit_terms     = cfg.fit_terms,
        nerr_terms    = cfg.nerr_terms,
        ff_shift      = cfg.ff_shift,
        use_error_jet = cfg.use_error_jet,
        name_prefix   = cfg.name_prefix,
        save_path     = cfg.save_path,
        write_summary = cfg.write_summary,
        use_cuda      = cfg.use_cuda,
        real_type     = T,
    )
end

end  # module Runner