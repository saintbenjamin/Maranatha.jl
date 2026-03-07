# ============================================================================
# src/Runner/Runner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Runner

import ..Utils.JobLoggerTools
import ..Utils.MaranathaIO
import ..Quadrature.QuadratureDispatch
import ..ErrorEstimate.ErrorDispatch
import ..LeastChiSquareFit

export run_Maranatha

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

Run a multi-resolution quadrature study, estimate an error scale at each resolution,
and return the raw convergence data needed for later ``h \\to 0`` extrapolation.

This is typically the **first step** in a standard `Maranatha.jl` workflow:
first call `run_Maranatha`, then pass its output to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref), and optionally visualize
the fitted result with [`Maranatha.PlotTools.plot_convergence_result`](@ref).

# Function description

`run_Maranatha` is the high-level entry point for generating a convergence dataset from
repeated quadrature evaluations at multiple resolutions.

It orchestrates the first half of the `Maranatha.jl` workflow by combining:

- [`Maranatha.Quadrature`](@ref):
  tensor-product quadrature dispatch (Newton-Cotes / Gauss / B-spline backends)
- [`Maranatha.ErrorEstimate`](@ref):
  residual-based derivative error-scale modeling based on midpoint residual expansions

The resulting dataset is then intended to be passed downstream to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
for weighted ``h \\to 0`` extrapolation.

For each resolution `N` in `nsamples`, the runner performs:

1. Compute the step size
   ``\\displaystyle{h = \\frac{b-a}{N}}``.

2. Evaluate the quadrature estimate via
   [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref).

3. Estimate the integration error scale via
   [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)
   or
   [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate_threads`](@ref),
   depending on `use_threads`.

4. Accumulate the triplets `(h, estimate, error_info)`.

After processing all requested resolutions, this function returns the collected
raw convergence-study data as a single `NamedTuple`.

In a typical workflow, the returned object is used immediately as input to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref),
for example through the fields
`result.a`, `result.b`, `result.h`, `result.avg`, and `result.err`.

This runner **does not perform the least-``\\chi^2`` fit itself**;
it is responsible only for dataset generation (and optional saving).

If `save_path` is provided, the returned result structure is also written to disk using
[`Maranatha.Utils.MaranathaIO.save_datapoint_results`](@ref).

The output filename has the form

```julia
result_\$(name_prefix)_\$(rule)_\$(boundary)_N_\$(nsamples[1])_N_\$(nsamples[end]).jld2
```

and is created inside the specified `save_path` directory.
If `write_summary = true`, a TOML summary file is written alongside the JLD2 file.

The fitting-related keywords `fit_terms` and `ff_shift` are stored in the returned result
for downstream convenience, but they are **not used internally** by `run_Maranatha`
to perform fitting.

# Arguments

* `integrand`:
  Callable integrand. May be a function, closure, or callable struct.
  It must accept `dim` scalar positional arguments.

* `a`, `b`:
  Scalar bounds defining the hypercube domain `[a,b]^n`,
  where `n` is the dimensionality.

# Keyword arguments

* `dim::Int = 1`:
  Dimensionality of the tensor-product quadrature.
  Internally dispatched through
  [`Maranatha.Quadrature.QuadratureDispatch.quadrature`](@ref),
  which supports specialized implementations for low dimensions and a general fallback.

* `nsamples = [2, 3, 4, 5, 6, 7, 8, 9]`:
  Vector of subdivision counts `N`.
  Each value defines a different resolution used in the convergence study.

* `rule::Symbol = :gauss_p4`:
  Quadrature rule identifier forwarded to integration and error estimation.
  Examples include Newton-Cotes, Gauss-family, and B-spline rule symbols
  supported by [`Maranatha.Quadrature`](@ref).

* `boundary::Symbol = :LU_EXEX`:
  Boundary pattern for the composite rule assembly.
  This is forwarded consistently to integration and error estimation.

* `err_method::Symbol = :forwarddiff`:
  Derivative backend used by the error-estimation dispatcher.
  Supported values currently include
  `:forwarddiff`, `:taylorseries`, `:fastdifferentiation`, and `:enzyme`.

* `fit_terms::Int = 4`:
  Suggested number of basis terms for a later least-``\\chi^2`` fit.
  This value is stored in the returned result so that downstream fitting code can reuse
  the same configuration without re-entering it manually.

* `nerr_terms::Int = 3`:
  Number of midpoint residual terms used by the derivative-based error estimator.

  * `1`  uses LO only
  * `>1` uses LO + NLO + ... up to `nerr_terms` terms
    (subject to the residual scan limit)

* `ff_shift::Int = 0`:
  Suggested forward shift for later fit-power selection inside
  [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
  Like `fit_terms`, this is stored in the returned result for downstream reuse rather than
  being applied internally by `run_Maranatha`.

* `use_threads::Bool = false`:
  If `true`, dispatches to the threaded error-estimation backend.

* `name_prefix::String = "Maranatha"`:
  Prefix used when constructing the output filename if results are written to disk.

* `save_path::Union{Nothing,AbstractString} = nothing`:
  Optional directory path where the runner writes the result file.
  If `nothing`, no file is written.

* `write_summary::Bool = true`:
  If `true`, a companion TOML summary file is written together with the JLD2 result file.

# Typical next step

A common downstream pattern is:

1. call `run_Maranatha(...)` to generate `result`
2. call [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref) using
   `result.a`, `result.b`, `result.h`, `result.avg`, and `result.err`
3. optionally call [`Maranatha.PlotTools.plot_convergence_result`](@ref)
   to visualize the fitted convergence curve and uncertainty band

# Returns

A single `NamedTuple` containing the raw convergence-study data and associated metadata:

* `a`, `b`:
  Integration bounds.

* `h`:
  Step sizes corresponding to the tested resolutions.

* `avg`:
  Quadrature estimates `I(h)`.

* `err`:
  Error-estimator outputs (NamedTuples returned by
  [`Maranatha.ErrorEstimate.ErrorDispatch.error_estimate`](@ref)
  or its threaded variant).

* `rule`, `boundary`:
  Rule configuration used during execution.

* `dim`:
  Dimensionality of the tensor-product quadrature.

* `err_method`:
  Derivative backend used for error estimation.

* `nerr_terms`:
  Number of midpoint residual terms included in the error model.

* `fit_terms`, `ff_shift`:
  Suggested fitting parameters stored for downstream use.

* `use_threads`:
  Indicates whether threaded error estimation was used.

# Design notes

* The runner is **dimension-agnostic** and supports arbitrary `n \\ge 1`
  subject to computational cost.
* Error estimators provide a *scale model* rather than a strict truncation bound,
  enabling stable weighted fits downstream.
* Logging and timing are centralized through
  [`Maranatha.Utils.JobLoggerTools`](@ref).
* Threaded error estimation is optionally enabled via `use_threads`,
  without changing the returned result structure.

# Example

The example below demonstrates the standard three-step workflow:
generate a convergence dataset, perform the downstream fit, and plot the resulting convergence curve.


```julia
f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```

"""
function run_Maranatha(
    integrand,
    a,
    b;
    dim=1,
    nsamples=[4,8,12,16],
    rule=:newton_p3,
    boundary=:LU_ININ,
    err_method::Symbol = :forwarddiff,  # :forwarddiff | :taylorseries | :fastdifferentiation | :enzyme
    fit_terms::Int = 2,
    nerr_terms::Int = 1,
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
        save_jld2_path = joinpath(
            save_path,
            "result_$(name_prefix)_$(rule)_$(boundary)_N_$(nsamples[1])_N_$(nsamples[end]).jld2"
        )
        MaranathaIO.save_datapoint_results(
            save_jld2_path,
            result;
            write_summary = write_summary,
        )
    end

    return result
end

end  # module Runner