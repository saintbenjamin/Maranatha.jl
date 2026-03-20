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
        result,
        fit_result;
        name::String = "Maranatha",
        out_dir::String = ".",
        format::Symbol = :tex,
        save_file::Bool = true,
        nerr_terms::Union{Nothing,Int} = nothing,
    ) -> String

Generate a convergence summary report in
[``\\LaTeX``](https://www.latex-project.org/) or Markdown format directly from a
structured quadrature result object and a fitted extrapolation result.

# Function description

This routine processes quadrature results at multiple resolutions, combines
them with a fitted extrapolation model, and produces a formatted report
summarizing:

- run configuration,
- raw estimates and uncertainties,
- extrapolated value,
- fit parameters and diagnostics.

Invalid or non-finite datapoints are automatically filtered before report
generation.

It allows direct reporting from a stored or freshly computed Maranatha result
object without manually unpacking arrays.

The uncertainty inputs may come from either residual-based error estimators or
refinement-based error estimators, as long as each error entry provides a
usable scalar uncertainty field.

# Arguments

- `result`:
  Object containing quadrature outputs and metadata, expected to expose fields
  such as `a`, `b`, `h`, `avg`, `err`, `fit_terms`, `nerr_terms`, `rule`, and
  `boundary`.

  The summary is generated from:

  - `result.a`, `result.b` as the integration-domain description,
  - `result.h` as the scalar step-size sequence,
  - `result.avg` as the corresponding integral estimates,
  - `result.err` as the error objects containing pointwise uncertainties,
  - `result.rule` and `result.boundary` as reporting labels.

  In rectangular-domain workflows, `result.h` is expected to be the scalarized
  step-size sequence used for fitting, plotting, and reporting, while any
  original per-axis step data remain outside this helper.

- `fit_result`:
  Object containing fit outputs.

# Keyword arguments

- `name::String = "Maranatha"`:
  Title or identifier used in the generated summary.
- `out_dir::String = "."`:
  Output directory for generated files when `save_file = true`.
- `format::Symbol = :tex`:
  Output format (`:tex` or `:md`).
- `save_file::Bool = true`:
  If `true`, write the generated summary to disk.
- `nerr_terms::Union{Nothing,Int} = nothing`:
  Optional override for the number of error-model terms used in the summary.
  If omitted, `result.nerr_terms` is used.

  When the error estimation method is `:refinement`, this value is ignored
  and automatically forced to `0`, since refinement-based estimates do not
  use an explicit error-model expansion.

# Returns

- `String`:
  The generated report text.

# Errors

- Throws an error via [`JobLoggerTools.error_benji`](@ref) if inputs are
  inconsistent or required fit information is missing.

# Notes

- Data are sorted from coarse to fine resolution (largest `h` first).
- Non-finite values are excluded automatically.
- This routine accepts both residual-based and refinement-based error-info
  objects, provided that each entry exposes either `.total` or `.estimate`.
- This wrapper is intended for convenience when working directly with the
  result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).
- For rectangular-domain workflows, the generated summary is based on the
  scalarized step-size sequence `result.h` rather than the original per-axis
  step tuples.
"""
function write_convergence_summary(
    result,
    fit_result;
    name::String = "Maranatha",
    out_dir::String = ".",
    format::Symbol = :tex,
    save_file::Bool = true,
    nerr_terms::Union{Nothing,Int} = nothing,
)
    a = result.a
    b = result.b
    hs = Vector{Float64}(result.h)
    estimates = Vector{Float64}(result.avg)
    errors = result.err
    rule = result.rule
    boundary = result.boundary
    err_method = result.err_method
    fit_terms = result.fit_terms
    nerr_terms_eff_input = isnothing(nerr_terms) ? result.nerr_terms : nerr_terms
    nerr_terms_eff = (err_method == :refinement) ? 0 : nerr_terms_eff_input

    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    hasproperty(fit_result, :powers) || JobLoggerTools.error_benji(
        "fit_result missing :powers"
    )
    hasproperty(fit_result, :params) || JobLoggerTools.error_benji(
        "fit_result missing :params"
    )
    hasproperty(fit_result, :cov) || JobLoggerTools.error_benji(
        "fit_result missing :cov"
    )
    hasproperty(fit_result, :estimate) || JobLoggerTools.error_benji(
        "fit_result missing :estimate"
    )
    hasproperty(fit_result, :error_estimate) || JobLoggerTools.error_benji(
        "fit_result missing :error_estimate"
    )

    fit_powers = fit_result.powers
    length(fit_powers) >= 2 || JobLoggerTools.error_benji(
        "fit_result.powers must contain at least constant and leading power"
    )

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

    isempty(hxp) && JobLoggerTools.error_benji(
        "No valid datapoints remain after filtering."
    )

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
        nerr_terms_eff,
    )

    if format == :tex
        text = _build_convergence_summary_tex(
            a, 
            b, 
            name, 
            hsp, 
            hxp, 
            estp, 
            errp,
            pvec, 
            λerr, 
            fit_powers, 
            I0, 
            I0_err, 
            red, 
            nerr_terms_eff_input;
            rule=rule, 
            boundary=boundary
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
            a, 
            b, 
            name, 
            hsp, 
            hxp, 
            estp, 
            errp,
            pvec, 
            λerr, 
            fit_powers, 
            I0, 
            I0_err, 
            red, 
            nerr_terms_eff_input;
            rule=rule, 
            boundary=boundary
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