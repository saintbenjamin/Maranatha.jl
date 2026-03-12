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
2. attach derivative-informed error-scale estimates,
3. collect the results into a uniform `NamedTuple`,
4. optionally save the dataset to disk for later reuse.

The main entry point is [`run_Maranatha`](@ref).
For least-``\\chi^2`` extrapolation of the generated dataset, see
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
For visualization of the resulting convergence behavior, see
[`Maranatha.PlotTools.plot_convergence_result`](@ref).
"""
module Runner

import ..TOML

import ..Utils.JobLoggerTools
import ..Utils.MaranathaIO
import ..Utils.MaranathaTOML
import ..Quadrature.QuadratureDispatch
import ..ErrorEstimate.ErrorDispatch

"""
    run_Maranatha(
        integrand,
        a,
        b;
        dim::Int = 1,
        nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
        rule::Symbol = :gauss_p4,
        boundary::Symbol = :LU_EXEX,
        err_method::Symbol = :forwarddiff,
        fit_terms::Int = 4,
        nerr_terms::Int = 3,
        ff_shift::Int = 0,
        use_threads::Bool = false,
        name_prefix::String = "Maranatha",
        save_path::Union{Nothing,AbstractString} = nothing,
        write_summary::Bool = true
    )

Run a multi-resolution quadrature study, estimate an error scale at each
resolution, and return the raw convergence data needed for later
``h \\to 0`` extrapolation.

This is typically the first stage of a standard `Maranatha.jl` workflow:
use `run_Maranatha` to build a convergence dataset, then pass that dataset to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref), and optionally
visualize the result with
[`Maranatha.PlotTools.plot_convergence_result`](@ref).

# What this function does

For each subdivision count `N` in `nsamples`, this routine:

1. computes the step size ``\\displaystyle{h = \\frac{b-a}{N}}``,
2. evaluates the quadrature estimate via [`QuadratureDispatch.quadrature`](@ref),
3. evaluates a derivative-informed error-scale model via
   [`ErrorDispatch.error_estimate`](@ref) or
   [`ErrorDispatch.error_estimate_threads`](@ref),
4. records the resulting step size, estimate, and error information for later aggregation into the returned dataset.

The collected results are returned as a single `NamedTuple` and can also be
written to disk for later reuse.

# Arguments

* `integrand`:
  Callable integrand. It must accept `dim` scalar positional arguments.

* `a`, `b`:
  Scalar bounds defining the hypercube domain ``[a,b]^n``.

# Keyword arguments

* `dim::Int = 1`:
  Tensor-product quadrature dimension.

* `nsamples = [2, 3, 4, 5, 6, 7, 8, 9]`:
  Subdivision counts used to build the convergence dataset.

* `rule::Symbol = :gauss_p4`:
  Quadrature rule identifier forwarded to both quadrature and error estimation.

* `boundary::Symbol = :LU_EXEX`:
  Boundary pattern used consistently across the quadrature pipeline.

* `err_method::Symbol = :forwarddiff`:
  Derivative backend used by the error-estimation dispatcher.

* `fit_terms::Int = 4`:
  Suggested number of basis terms for a later least-``\\chi^2`` fit.
  This value is stored in the returned result for downstream reuse.

* `nerr_terms::Int = 3`:
  Number of midpoint-residual terms used in the derivative-based error model.

* `ff_shift::Int = 0`:
  Suggested forward shift for downstream fit-power selection.
  This value is stored in the returned result but is not applied here.

* `use_threads::Bool = false`:
  If `true`, use the threaded error-estimation backend.

* `name_prefix::String = "Maranatha"`:
  Prefix used when constructing output filenames.

* `save_path::Union{Nothing,AbstractString} = nothing`:
  Output directory for optional result saving. If `nothing`, no file is written.

* `write_summary::Bool = true`:
  If `true`, write a [`TOML`](https://toml.io/en/) summary alongside the saved `JLD2` dataset.

# Returns

A `NamedTuple` containing the raw convergence-study dataset and associated metadata.
Important fields include:

* `a`, `b`: integration bounds,
* `h`: step sizes,
* `avg`: quadrature estimates,
* `err`: error-estimator outputs,
* `rule`, `boundary`, `dim`, `err_method`: execution metadata,
* `nerr_terms`, `fit_terms`, `ff_shift`, `use_threads`: downstream configuration hints.

# Saving behavior

If `save_path` is provided, the result is written using
[`MaranathaIO.save_datapoint_results`](@ref).
The output filename has the form

```julia
result_\$(name_prefix)_\$(rule)_\$(boundary)_N_\$(join(sort(nsamples), "_")).jld2
```

If `write_summary = true`, a [`TOML`](https://toml.io/en/) summary file is written alongside it.

# Notes

* `run_Maranatha` generates datasets only; it does not perform fitting.
* The error object is an error-scale model intended for stable downstream weighting,
  not necessarily a strict truncation bound.
* `fit_terms` and `ff_shift` are preserved for convenience so that later fitting code
  can reuse the same workflow settings.

# Examples

Direct function call:

```julia
using Maranatha

f(x) = sin(x)

run_result = run_Maranatha(
    f,
    0.0,
    pi;
    dim = 1,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
)
```

Configuration-file workflow:

```julia
using Maranatha

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
    dim=1,
    nsamples=[2, 3, 4, 5, 6, 7, 8, 9],
    rule=:gauss_p4,
    boundary=:LU_EXEX,
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    fit_terms::Int = 4,
    nerr_terms::Int = 3,
    ff_shift::Int = 0,
    use_threads::Bool = false,
    name_prefix::String = "Maranatha",
    save_path::Union{Nothing,AbstractString}=nothing,
    write_summary::Bool=true,
)
    estimates = Float64[]      # List of quadrature results
    error_infos = NamedTuple[] # List of estimated error infos
    hs = Float64[]             # List of step sizes

    for N in nsamples
        h = (b - a) / N
        push!(hs, h)

        I = QuadratureDispatch.quadrature(
            integrand,
            a,
            b,
            N,
            dim,
            rule,
            boundary
        )

        err = if use_threads
            ErrorDispatch.error_estimate_threads(
                integrand,
                a,
                b,
                N,
                dim,
                rule,
                boundary;
                err_method=err_method,
                nerr_terms=nerr_terms
            )
        else
            ErrorDispatch.error_estimate(
                integrand,
                a,
                b,
                N,
                dim,
                rule,
                boundary;
                err_method=err_method,
                nerr_terms=nerr_terms
            )
        end

        push!(estimates, I)
        push!(error_infos, err)
    end

    result = (;
        a           = a,
        b           = b,
        h           = hs,
        avg         = estimates,
        err         = error_infos,
        rule        = rule,
        boundary    = boundary,
        dim         = dim,
        err_method  = err_method,
        nerr_terms  = nerr_terms,
        fit_terms   = fit_terms,
        ff_shift    = ff_shift,
        use_threads = use_threads,
    )

    if save_path !== nothing
        Nstr = join(sort(nsamples), "_")
        save_jld2_path = joinpath(
            save_path,
            "result_$(name_prefix)_$(rule)_$(boundary)_N_$(Nstr).jld2"
        )
        MaranathaIO.save_datapoint_results(
            save_jld2_path,
            result;
            write_summary = write_summary,
        )
    end

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
        use_threads   = cfg.use_threads,
        name_prefix   = cfg.name_prefix,
        save_path     = cfg.save_path,
        write_summary = cfg.write_summary,
    )
end

end  # module Runner