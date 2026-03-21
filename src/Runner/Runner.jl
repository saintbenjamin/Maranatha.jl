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
For least-``\\chi^2`` extrapolation of the generated dataset, see
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
For visualization of the resulting convergence behavior, see
[`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref).
"""
module Runner

import ..DoubleFloats
import ..TOML

import ..Utils.JobLoggerTools
import ..Utils.MaranathaIO
import ..Utils.MaranathaTOML
import ..Utils.QuadratureBoundarySpec
import ..Quadrature.QuadratureRuleSpec
import ..Quadrature.QuadratureDispatch
import ..Quadrature.NewtonCotes
import ..ErrorEstimate.ErrorDispatch

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

At the beginning of the run, this routine clears the global
derivative-based error-estimation caches via
[`ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!`](@ref)
when a derivative-based backend is selected.
The refinement-based path does not use these caches.

For each subdivision count `N` in `nsamples`, this routine:

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
    `:fastdifferentiation`, `:enzyme`) selects a derivative-based estimator.

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
  and for tuple-based axis-wise domains this is the L2 norm of the per-axis
  step tuple,
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
    T = isnothing(real_type) ? Float64 : real_type

    if use_cuda && !(T === Float32 || T === Float64)
        throw(ArgumentError(
            "CUDA mode currently supports only Float32 or Float64 real_type (got $(T))."
        ))
    end

    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    if err_method === :refinement
        QuadratureRuleSpec._common_rule_family(rule, dim)
    end

    JobLoggerTools.println_benji(
        "run config: dim=$(dim), rule=$(string(rule)), boundary=$(string(boundary)), err_method=$(err_method)"
    )

    # ------------------------------------------------------------
    # Domain handling
    # ------------------------------------------------------------
    is_rect_domain = a isa AbstractVector || a isa Tuple

    if !is_rect_domain
        aT = convert(T, a)
        bT = convert(T, b)
    else
        length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
        length(b) == dim || throw(ArgumentError("length(b) must equal dim"))

        aT = a isa Tuple ? ntuple(i -> convert(T, a[i]), dim) : T[convert(T, a[i]) for i in 1:dim]
        bT = b isa Tuple ? ntuple(i -> convert(T, b[i]), dim) : T[convert(T, b[i]) for i in 1:dim]
    end

    threaded_subgrid = (!use_cuda) && (Base.Threads.nthreads() > 1)

    if err_method !== :refinement
        ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!()
    end

    estimates = T[]
    error_infos = NamedTuple[]
    hs    = Vector{Any}()
    hs_l2 = Vector{T}()

    nsamples = _sanitize_run_nsamples(
        nsamples,
        rule,
        boundary,
        dim,
    )

    for N in nsamples
        h = if !is_rect_domain
            (bT - aT) / T(N)
        else
            aT isa Tuple ?
                ntuple(i -> (bT[i] - aT[i]) / T(N), dim) :
                T[(bT[i] - aT[i]) / T(N) for i in 1:dim]
        end

        push!(hs, h)

        # --- L2 norm scalar h ---
        h_l2 = h isa Tuple ?
            sqrt(sum(x -> x*x, h)) :
            h

        push!(hs_l2, h_l2)

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

        push!(estimates, I)
        push!(error_infos, err)
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

    if save_path !== nothing
        save_path_abs = isabspath(save_path) ? save_path : joinpath(pwd(), save_path)

        mkpath(save_path_abs)

        Nstr = join(sort(nsamples), "_")
        spec_str = MaranathaIO._rule_boundary_filename_token(aT, bT, rule, boundary)

        save_jld2_path = joinpath(
            save_path_abs,
            "result_$(name_prefix)_$(spec_str)_N_$(Nstr).jld2"
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

    T = cfg.real_type === :Float32  ? Float32  :
        cfg.real_type === :Float64  ? Float64  :
        cfg.real_type === :BigFloat ? BigFloat :
        cfg.real_type === :Double64 ? DoubleFloats.Double64 :
        error("Unsupported real_type=$(cfg.real_type)")

    # # --- Domain casting to selected real type T ---
    # _cast_domain(x) =
    #     x isa Tuple           ? Tuple(T(v) for v in x) :
    #     x isa AbstractVector  ? [T(v) for v in x]      :
    #                             T(x)

    # aT = _cast_domain(cfg.a)
    # bT = _cast_domain(cfg.b)

    return Base.invokelatest(
        run_Maranatha,
        integrand,
        cfg.a, # aT,
        cfg.b; # bT;
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

"""
    _sanitize_run_nsamples(
        nsamples::Vector{Int},
        rule,
        boundary,
        dim::Int = 1,
    ) -> Vector{Int}

Sanitize a candidate subdivision sequence for Newton-Cotes composite rules.

# Function description
Newton-Cotes composite formulas do not accept arbitrary subdivision counts.
For a given local node count `p` and boundary pattern `boundary`, valid values
must satisfy the tiling constraint

```math
N_{\\mathrm{sub}} = w_L + m (p-1) + w_R,
```

where `w_L` and `w_R` are the boundary block widths and `m ≥ 0` is an integer.

When `rule` and `boundary` are axis-wise specifications, this helper computes a
single subdivision sequence that is simultaneously valid for every
Newton-Cotes-active axis. Axes that use non-Newton-Cotes rule families are
ignored for this admissibility check.

This helper transforms an arbitrary input sequence into a valid sequence that:

* preserves the original length,
* forms a valid arithmetic progression with step `(p - 1)`,
* starts from the nearest admissible value not exceeding the first input
  element, or from the smallest admissible value if none is smaller.

If `rule` is not a Newton-Cotes rule, the input is returned unchanged.

# Arguments

* `nsamples::Vector{Int}`
  Candidate subdivision counts supplied by the caller.

* `rule`
  Quadrature rule specification. This may be either a scalar rule symbol shared
  across all axes or a tuple / vector of rule symbols of length `dim`.

* `boundary`
  Boundary specification. This may be either a scalar boundary symbol shared
  across all axes or a tuple / vector of boundary symbols of length `dim`.

* `dim::Int = 1`
  Problem dimension used when resolving axis-wise `rule` / `boundary`
  specifications.

# Returns

* `Vector{Int}`
  A corrected subdivision sequence compatible with the Newton-Cotes composite
  tiling constraint. The returned vector has the same length as `nsamples`.

# Errors

* Propagates validation errors from rule and boundary specification checks, and
  from the Newton-Cotes helper routines used to determine admissible
  subdivision counts.
* Does not throw for inadmissible subdivision counts; instead, it adjusts them.

# Notes

* This helper is intended for internal use by runner-level components that
  accept user-supplied subdivision arrays.
* A warning is emitted if the sequence is modified.
* The resulting sequence always represents a monotone refinement ladder whose
  common step is the least common multiple of all active Newton-Cotes block
  widths.
"""
function _sanitize_run_nsamples(
    nsamples::Vector{Int},
    rule,
    boundary,
    dim::Int = 1,
)::Vector{Int}

    isempty(nsamples) && return nsamples

    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    rule_axes = [QuadratureRuleSpec._rule_at(rule, d, dim) for d in 1:dim]
    nc_axes = [d for d in 1:dim if NewtonCotes._is_newton_cotes_rule(rule_axes[d])]

    isempty(nc_axes) && return nsamples

    function _is_valid_common_N(Ncand::Int)::Bool
        for d in nc_axes
            rd = rule_axes[d]
            bd = QuadratureBoundarySpec._boundary_at(boundary, d, dim)
            p = NewtonCotes._parse_newton_p(rd)
            Nd = NewtonCotes._nearest_valid_Nsub(p, bd, Ncand)
            Nd == Ncand || return false
        end
        return true
    end

    steps = Int[Quadrature.NewtonCotes._parse_newton_p(rule_axes[d]) - 1 for d in nc_axes]
    step = foldl(lcm, steps; init = 1)

    N0 = first(nsamples)
    start = nothing

    for Ncand in N0:-1:1
        if _is_valid_common_N(Ncand)
            start = Ncand
            break
        end
    end

    if isnothing(start)
        Ncand = 1
        while true
            if _is_valid_common_N(Ncand)
                start = Ncand
                break
            end
            Ncand += 1
        end
    end

    newN = [start + (i - 1) * step for i in 1:length(nsamples)]

    if newN != nsamples
        JobLoggerTools.warn_benji(
            "nsamples corrected for rule=$(rule), boundary=$(boundary), dim=$(dim)\n" *
            "input = $(nsamples)\n" *
            "using = $(newN)"
        )
    end

    return newN
end

end  # module Runner