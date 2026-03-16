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
import ..ErrorEstimate.ErrorDispatchRefine
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
        err_method::Symbol = :refinement,
        fit_terms::Int = 4,
        nerr_terms::Int = 3,
        ff_shift::Int = 0,
        use_error_jet::Bool = false,
        name_prefix::String = "Maranatha",
        save_path::Union{Nothing,AbstractString} = nothing,
        write_summary::Bool = true
    )

Run a multi-resolution quadrature study, estimate an error scale at each resolution 
using either a derivative-based or refinement-based backend, 
and return the raw convergence data needed for later ``h \\to 0`` extrapolation.


This is typically the first stage of a standard `Maranatha.jl` workflow:
use `run_Maranatha` to build a convergence dataset, then pass that dataset to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref), and optionally
visualize the result with
[`Maranatha.PlotTools.plot_convergence_result`](@ref).

# What this function does

At the beginning of the run, this routine clears the global derivative-based error-estimation caches via 
[`ErrorDispatch.clear_error_estimate_caches!`](@ref) when a derivative-based backend is selected. 
The refinement-based path does not use those caches.

For each subdivision count `N` in `nsamples`, this routine:

1. computes the step size ``\\displaystyle{h = \\frac{b-a}{N}}``,
2. evaluates the quadrature estimate via [`QuadratureDispatch.quadrature`](@ref),
3. evaluates either a derivative-informed or refinement-based error-scale model,
4. records the resulting step size, estimate, and error information for later
   aggregation into the returned dataset.

The collected results are returned as a single `NamedTuple` and can also be
written to disk for later reuse.

# Current error-estimation path

The active error-estimation path depends first on `err_method`, and then on `use_error_jet`.

- If `err_method == :refinement`, this routine calls [`ErrorDispatchRefine.error_estimate_refine`](@ref). 
  In this path, `use_error_jet` is ignored because refinement-based estimation does not use derivative jets.

- If `err_method != :refinement` and `use_error_jet = true`, this routine calls [`ErrorDispatch.error_estimate_jet`](@ref).

- If `err_method != :refinement` and `use_error_jet = false`, this routine calls [`ErrorDispatch.error_estimate`](@ref).

Thus, the refinement backend is selected explicitly through `err_method = :refinement`, while `use_error_jet` only affects the derivative-based branch.

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

* `err_method::Symbol = :refinement`: Error-estimation backend selector. If set to `:refinement`, use the refinement-based dispatcher [`ErrorDispatchRefine.error_estimate_refine`](@ref). Otherwise, use the derivative-based dispatcher with the requested derivative backend such as `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, or `:enzyme`.

* `fit_terms::Int = 4`:
  Suggested number of basis terms for a later least-``\\chi^2`` fit.
  This value is stored in the returned result for downstream reuse.

* `nerr_terms::Int = 3`:
  Number of midpoint-residual terms used in the derivative-based error model.

* `ff_shift::Int = 0`:
  Suggested forward shift for downstream fit-power selection.
  This value is stored in the returned result but is not applied here.

* `use_error_jet::Bool = false`: Controls only the derivative-based error-estimation branch. 
If `true`, use [`ErrorDispatch.error_estimate_jet`](@ref). 
If `false`, use [`ErrorDispatch.error_estimate`](@ref).
This option is ignored when `err_method == :refinement`.

* `name_prefix::String = "Maranatha"`:
  Prefix used when constructing output filenames.

* `save_path::Union{Nothing,AbstractString} = nothing`:
  Output directory for optional result saving. If `nothing`, no file is written.

* `write_summary::Bool = true`:
  If `true`, write a [`TOML`](https://toml.io/en/) summary alongside the saved `JLD2` dataset.

# Returns

A `NamedTuple` containing the raw convergence-study dataset and associated metadata.
The `err` field may therefore contain either derivative-based error objects 
or refinement-based error objects, depending on `err_method`.
Important fields include:

* `a`, `b`: integration bounds,
* `h`: step sizes,
* `avg`: quadrature estimates,
* `err`: error-estimator outputs,
* `rule`, `boundary`, `dim`, `err_method`: execution metadata,
* `nerr_terms`, `fit_terms`, `ff_shift`, `use_error_jet`: downstream configuration hints.

# Saving behavior

If `save_path` is provided, the result is written using
[`MaranathaIO.save_datapoint_results`](@ref).
Relative save paths are normalized against the current working directory before
directory creation and file writing.

The output filename has the form

```julia
result_\$(name_prefix)_\$(rule)_\$(boundary)_N_\$(join(sort(nsamples), "_")).jld2
```

If `write_summary = true`, a [`TOML`](https://toml.io/en/) summary file is written alongside it.

# Notes

* `run_Maranatha` generates datasets only; it does not perform fitting.
* The error object is an error-scale model intended for stable downstream weighting,
not necessarily a strict truncation bound. Depending on `err_method`, it may come either from a derivative-based asymptotic model or from a refinement-based coarse-versus-fine quadrature difference.
  not necessarily a strict truncation bound.
* `fit_terms` and `ff_shift` are preserved for convenience so that later fitting code
  can reuse the same workflow settings.
* The derivative-based and refinement-based branches now coexist in the public API.
The refinement branch is selected explicitly with `err_method = :refinement`, while `use_error_jet` only affects the derivative-based branch.

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
    err_method = :refinement,
)
```
To use the refinement-based estimator instead, pass `err_method = :refinement`.

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
    err_method::Symbol = :refinement,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    fit_terms::Int = 4,
    nerr_terms::Int = 3,
    ff_shift::Int = 0,
    use_error_jet::Bool = false,
    name_prefix::String = "Maranatha",
    save_path::Union{Nothing,AbstractString}=nothing,
    write_summary::Bool=true,
)

    JobLoggerTools.log_stage_benji("Start run_Maranatha")

    if err_method !== :refinement
        ErrorDispatch.clear_error_estimate_caches!()
    end
    
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

        err = if err_method === :refinement
            ErrorDispatchRefine.error_estimate_refine(
                integrand,
                a,
                b,
                N,
                dim,
                rule,
                boundary
            )
        else
            if use_error_jet
                ErrorDispatch.error_estimate_jet(
                    integrand,
                    a,
                    b,
                    N,
                    dim,
                    rule,
                    boundary;
                    err_method = err_method,
                    nerr_terms = nerr_terms
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
                    err_method = err_method,
                    nerr_terms = nerr_terms
                )
            end
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
        use_error_jet = use_error_jet,
    )

    if save_path !== nothing
        # ---- normalize to current working directory ----
        save_path_abs = isabspath(save_path) ? save_path : joinpath(pwd(), save_path)

        mkpath(save_path_abs)

        Nstr = join(sort(nsamples), "_")
        save_jld2_path = joinpath(
            save_path_abs,
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
        use_error_jet   = cfg.use_error_jet,
        name_prefix   = cfg.name_prefix,
        save_path     = cfg.save_path,
        write_summary = cfg.write_summary,
    )
end

end  # module Runner