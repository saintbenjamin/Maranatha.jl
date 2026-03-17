# ============================================================================
# src/Documentation/Reporter/write_convergence_summary.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    write_convergence_summary(
        a, b, name, hs, estimates, errors,
        fit_terms, nerr_terms, fit_result;
        rule, boundary, out_dir, format, save_file
    ) -> String

Generate a convergence summary report in [``\\LaTeX``](https://www.latex-project.org/) or Markdown format.

# Function description

This routine processes quadrature results at multiple resolutions,
combines them with a fitted extrapolation model, and produces a formatted
report summarizing:

- run configuration,
- raw estimates and uncertainties,
- extrapolated value,
- fit parameters and diagnostics.

Invalid or non-finite datapoints are automatically filtered before
report generation.

The uncertainty inputs may come from either residual-based error estimators or
refinement-based error estimators, as long as each error entry provides a
usable scalar uncertainty field.

# Arguments

- `a`, `b`: Integration interval endpoints.
- `name`: Identifier for the experiment or integrand.
- `hs`: Step sizes used in the quadrature study.
- `estimates`: Corresponding integral estimates.
- `errors`: Error objects containing pointwise uncertainties.

  Each entry is expected to provide either:

  - a `.total` field, as in the residual-based error-estimation workflow, or
  - an `.estimate` field, as in the refinement-based error-estimation workflow.

  The reporting routine converts each entry into a nonnegative scalar
  uncertainty through an internal extractor.
- `fit_terms`: Number of fit parameters used.
- `nerr_terms`: Number of error terms in the model.
- `fit_result`: Object containing fit outputs.

# Keyword arguments

- `rule`: Quadrature rule (default `:gauss_p3`).
- `boundary`: Boundary scheme (default `:LU_ININ`).
- `out_dir`: Output directory.
- `format`: Output format (`:tex` or `:md`).
- `save_file`: Whether to write the file to disk.

# Returns

- `String`: The generated report text.

# Errors

- Throws an error via [`JobLoggerTools.error_benji`](@ref) if inputs are inconsistent
  or required fit information is missing.

# Notes

- Data are sorted from coarse to fine resolution (largest `h` first).
- Non-finite values are excluded automatically.
- This routine accepts both residual-based and refinement-based error-info
  objects, provided that each entry exposes either `.total` or `.estimate`.
"""
function write_convergence_summary(
    a::Real,
    b::Real,
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector,
    fit_terms::Int,
    nerr_terms::Int,
    fit_result;
    rule::Symbol = :gauss_p3,
    boundary::Symbol = :LU_ININ,
    out_dir::String = ".",
    format::Symbol = :tex,
    save_file::Bool = true,
)
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    hasproperty(fit_result, :powers)  || JobLoggerTools.error_benji("fit_result missing :powers")
    hasproperty(fit_result, :params)  || JobLoggerTools.error_benji("fit_result missing :params")
    hasproperty(fit_result, :cov)     || JobLoggerTools.error_benji("fit_result missing :cov")
    hasproperty(fit_result, :estimate) || JobLoggerTools.error_benji("fit_result missing :estimate")
    hasproperty(fit_result, :error_estimate) || JobLoggerTools.error_benji("fit_result missing :error_estimate")

    fit_powers = fit_result.powers
    length(fit_powers) >= 2 || JobLoggerTools.error_benji("fit_result.powers must contain at least constant and leading power")

    # Support both residual-based (.total) and refinement-based (.estimate) error objects.
    @inline function _extract_error_total(e)
        if hasproperty(e, :total)
            return float(e.total)
        elseif hasproperty(e, :estimate)
            return abs(float(e.estimate))
        else
            JobLoggerTools.error_benji(
                "Unsupported error-info structure (need :total or :estimate)."
            )
        end
    end

    lead_pow = fit_powers[2]

    hx = hs .^ lead_pow
    errvals = [_extract_error_total(e) for e in errors]

    mask = isfinite.(hs) .& isfinite.(hx) .& isfinite.(estimates) .& isfinite.(errvals)
    hsp   = hs[mask]
    hxp   = hx[mask]
    estp  = estimates[mask]
    errp  = errvals[mask]

    isempty(hxp) && JobLoggerTools.error_benji("No valid datapoints remain after filtering.")

    # coarse -> fine ordering:
    # larger h first, smaller h last
    perm = sortperm(hsp; rev=true)
    hsp  = hsp[perm]
    hxp  = hxp[perm]
    estp = estp[perm]
    errp = errp[perm]

    pvec = fit_result.params
    CovS = LinearAlgebra.Symmetric(Matrix(fit_result.cov))
    length(pvec) == length(fit_powers) || JobLoggerTools.error_benji(
        "fit_result.powers length mismatch: expected $(length(pvec)), got $(length(fit_powers))"
    )

    λerr = Float64[]
    for i in eachindex(pvec)
        push!(λerr, sqrt(abs(CovS[i, i])))
    end

    I0     = fit_result.estimate
    I0_err = fit_result.error_estimate
    chisq  = hasproperty(fit_result, :chisq) ? fit_result.chisq : NaN
    dof    = hasproperty(fit_result, :dof)   ? fit_result.dof   : NaN
    red    = (isfinite(chisq) && isfinite(dof) && dof != 0) ? chisq / dof : NaN

    summary_basename = _build_convergence_summary_basename(
        name,
        rule,
        boundary,
        fit_terms,
        nerr_terms,
    )

    if format == :tex
        text = _build_convergence_summary_tex(
            a, b, name, hsp, hxp, estp, errp,
            pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
            rule=rule, boundary=boundary
        )
        ext = "tex"
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).$ext")
            open(outfile, "w") do io
                write(io, text)
            end
        end
        return text

    elseif format == :md
        text = _build_convergence_summary_md(
            a, b, name, hsp, hxp, estp, errp,
            pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
            rule=rule, boundary=boundary
        )
        ext = "md"
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).$ext")
            open(outfile, "w") do io
                write(io, text)
            end
        end
        return text

    else
        JobLoggerTools.error_benji("Unsupported format=$(format). Use :tex or :md.")
    end
end

"""
    write_convergence_summary(
        result,
        fit_result;
        name, rule, boundary, out_dir, format, save_file,
        nterms, nerr_terms
    ) -> String

Convenience wrapper for [`write_convergence_summary`](@ref) using a result object.

# Function description

This overload extracts required fields from a structured quadrature result
object and forwards them to the main reporting routine.

It allows direct reporting without manually unpacking the result structure.

In addition to forwarding the fit metadata stored in `result`, this wrapper
also allows the caller to override the fit-model settings used for summary
generation via the `nterms` and `nerr_terms` keyword arguments.

# Arguments

- `result`: Object containing quadrature outputs and metadata.
- `fit_result`: Extrapolation fit result.

# Keyword arguments

- `name::String = "Maranatha"`
  : Title or identifier used in the generated summary.

- `rule::Symbol = result.rule`
  : Quadrature rule label to display in the summary.

- `boundary::Symbol = result.boundary`
  : Boundary-condition label to display in the summary.

- `out_dir::String = "."`
  : Output directory for generated files when `save_file = true`.

- `format::Symbol = :tex`
  : Output format for the generated summary.

- `save_file::Bool = true`
  : If `true`, write the generated summary to disk.

- `nterms::Union{Nothing,Int} = nothing`
  : Optional override for the number of fit terms used in the summary.
    If `nothing`, `result.fit_terms` is used.

- `nerr_terms::Union{Nothing,Int} = nothing`
  : Optional override for the number of error-model terms used in the summary.
    If `nothing`, `result.nerr_terms` is used.

# Returns

- `String`: Generated report text.

# Notes

- The function assumes that `result` exposes fields compatible with the main
  reporting interface, including `a`, `b`, `h`, `avg`, `err`,
  `fit_terms`, `nerr_terms`, `rule`, and `boundary`.
- This wrapper is intended for convenience when working directly with the
  result object returned by [`Maranatha.Runner.run_Maranatha`](@ref), while
  still allowing limited report-level customization without manually
  reconstructing the full argument list.
"""
function write_convergence_summary(
    result,
    fit_result;
    name::String = "Maranatha",
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    out_dir::String = ".",
    format::Symbol = :tex,
    save_file::Bool = true,
    nterms::Union{Nothing,Int} = nothing,
    nerr_terms::Union{Nothing,Int} = nothing,    
)
    fit_nterms = isnothing(nterms) ? result.fit_terms : nterms
    fit_nerr_terms = isnothing(nerr_terms) ? result.nerr_terms : nerr_terms

    return write_convergence_summary(
        result.a,
        result.b,
        name,
        Vector{Float64}(result.h),
        Vector{Float64}(result.avg),
        result.err,
        fit_nterms,
        fit_nerr_terms,
        fit_result;
        rule = rule,
        boundary = boundary,
        out_dir = out_dir,
        format = format,
        save_file = save_file,
    )
end