# ============================================================================
# src/Utils/Reporter/report_summary.jl
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

# Arguments

- `a`, `b`: Integration interval endpoints.
- `name`: Identifier for the experiment or integrand.
- `hs`: Step sizes used in the quadrature study.
- `estimates`: Corresponding integral estimates.
- `errors`: Error objects containing total uncertainties.
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

    lead_pow = fit_powers[2]

    hx = hs .^ lead_pow
    errvals = [e.total for e in errors]

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

"""
    _build_convergence_summary_basename(
        name,
        rule,
        boundary,
        fit_terms,
        nerr_terms,
    ) -> String

Construct a standardized base filename for convergence-summary outputs.

# Function description

This helper builds a descriptive basename encoding key run parameters,
including the integrand name, quadrature rule, boundary condition,
and fitting configuration.

The result is typically used as the stem for output files such as:

- [``\\LaTeX``](https://www.latex-project.org/) summary tables
- Markdown reports
- auxiliary artifacts

Optional suffixes are appended only when the corresponding values are not
`nothing`.

# Arguments

- `name`: Identifier for the integrand or experiment.
- `rule`: Quadrature rule used.
- `boundary`: Boundary-handling scheme.
- `fit_terms`: Number of terms used in the extrapolation fit.
- `nerr_terms`: Number of error terms included in the model.

# Returns

- `String`: A filesystem-friendly basename.

# Notes

- The function performs no sanitization beyond string conversion.
- Callers are responsible for ensuring the result is valid as a filename.
"""
function _build_convergence_summary_basename(
    name::AbstractString,
    rule,
    boundary,
    fit_terms,
    nerr_terms,
)
    base = "summary_$(name)_$(String(rule))_$(String(boundary))"

    fit_terms_suffix = isnothing(fit_terms) ? "" : "_ff_$(fit_terms)"
    nerr_terms_suffix = isnothing(nerr_terms) ? "" : "_er_$(nerr_terms)"

    return base * fit_terms_suffix * nerr_terms_suffix
end

"""
    _build_convergence_summary_tex(
        a, b, name, hsp, hxp, estp, errp,
        pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
        rule, boundary
    ) -> String

Construct a [``\\LaTeX``](https://www.latex-project.org/) convergence-summary document fragment.

# Function description

This helper produces a set of [``\\LaTeX``](https://www.latex-project.org/) tables summarizing:

1. Fit parameters and goodness-of-fit statistics,
2. Run configuration,
3. Quadrature estimates across resolutions.

The output is intended for inclusion in larger [``\\LaTeX``](https://www.latex-project.org/) documents.

# Arguments

- `a`, `b`: Integration interval.
- `name`: Experiment identifier.
- `hsp`: Step sizes (filtered and ordered).
- `hxp`: Transformed step sizes (``h^p``).
- `estp`: Estimates at each resolution.
- `errp`: Corresponding uncertainties.
- `pvec`: Fit parameters.
- `λerr`: Parameter uncertainties.
- `fit_powers`: Model exponents.
- `I0`, `I0_err`: Extrapolated value and uncertainty.
- `red`: Reduced ``\\chi^2``.
- `nerr_terms`: Number of error terms.

# Keyword arguments

- `rule`, `boundary`: Run configuration metadata.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/) code fragment.

# Notes

- No preamble or document environment is included.
- Numeric formatting uses specialized helpers for readability.
"""
function _build_convergence_summary_tex(
    a, b, name, hsp, hxp, estp, errp,
    pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
    rule, boundary
)

    safe_name = _latex_escape_underscore(name)
    safe_rule = _latex_escape_underscore(String(rule))
    safe_boundary = _latex_escape_underscore(String(boundary))
    fit_model_tex = _build_fit_model_tex(fit_powers)

    io = IOBuffer()

    println(io, "% Auto-generated by write_convergence_summary")
    println(io, "")

    println(io, "\\begin{table}[ht!]")
    println(io, "\\begin{ruledtabular}")
    println(io, "\\caption{Least-\$\\chi^2\$ fit results using the model \$$(fit_model_tex)\$.}")
    println(io, "\\begin{tabular}{ @{\\quad} c @{\\quad} | @{\\quad} l @{\\quad} }")
    println(io, "parameter & \$\\hphantom{-}\$fit result \\\\")
    println(io, "\\hline")
    for i in eachindex(pvec)
        is_const = (fit_powers[i] == 0)

        λname = is_const ? raw"$\lambda_0$" : "\$\\lambda_$(i-1)\$"
        λtxt  = _fmt_avgerr_tex(pvec[i], λerr[i])

        # --- Align sign: add phantom minus if not negative ---
        aligned = startswith(λtxt, "-") ? λtxt : "\\hphantom{-}$λtxt"

        # --- Bold only for constant term ---
        valtex = is_const ? "\\mathbf{$aligned}" : aligned

        println(io, "$λname & \$$valtex\$ \\\\")
    end
    println(io, "\\hline")
    red_txt = _fmt_tex_texttt_sci(red)
    red_txt = startswith(red_txt, "-") ? red_txt : "\$\\hphantom{-}\$$red_txt"
    println(io, "\$\\chi^2/\\mathrm{d.o.f.}\$ & $red_txt \\\\")
    println(io, "\\end{tabular}")
    println(io, "\\end{ruledtabular}")
    println(io, "\\end{table}")
    println(io, "")

    println(io, "\\begin{table}[ht!]")
    println(io, "\\begin{ruledtabular}")
    println(io, "\\caption{Run configuration for \\texttt{$(safe_name)}}")

    println(io, "\\begin{tabular}{ @{\\quad} c @{\\quad} | @{\\quad} c @{\\quad} | @{\\quad} c @{\\quad} }")

    # --- Header row ---
    println(io,
        "Interval & Rule (Boundary) & Number of error terms \\\\"
    )
    println(io, "\\hline")

    # --- Value row ---
    println(io,
        "\$[$(a), $(b)]\$ & " *
        "\\texttt{$(safe_rule)} (\\texttt{$(safe_boundary)}) & " *
        "\$$(nerr_terms)\$ \\\\"
    )

    println(io, "\\end{tabular}")
    println(io, "\\end{ruledtabular}")
    println(io, "\\end{table}")
    println(io, "")

    println(io, "\\begin{table}[ht!]")
    println(io, "\\begin{ruledtabular}")
    println(io, "\\caption{Quadrature estimates and uncertainties for different step sizes}")
    println(io, "\\begin{tabular}{@{\\quad} l @{\\quad} l @{\\quad} | @{\\quad} l @{\\quad}}")
        println(io, "\$h\$ & \$h^{$(fit_powers[2])}\$ & \$\\hphantom{-}I(h)\$ \\\\")
    println(io, "\\hline")
    for i in eachindex(hsp)
        htxt  = _fmt_tex_texttt_sci(hsp[i])
        hptxt = _fmt_tex_texttt_sci(hxp[i])
        qtxt  = _fmt_avgerr_tex(estp[i], abs(errp[i]))

        # --- Align sign ---
        qtxt = startswith(qtxt, "-") ? qtxt : "\\hphantom{-}$qtxt"

        println(io, "$htxt & $hptxt & \$$qtxt\$ \\\\")
    end
    println(io, "\\hline")
    htxt  = _fmt_tex_texttt_sci(0.0)
    hptxt = _fmt_tex_texttt_sci(0.0)
    qtxt  = _fmt_avgerr_tex(I0, I0_err)
    qtxt = startswith(qtxt, "-") ? qtxt : "\\hphantom{-}$qtxt"
    println(io, "$htxt & $hptxt & \$\\mathbf{$qtxt}\$ \\\\")
    println(io, "\\end{tabular}")
    println(io, "\\end{ruledtabular}")
    println(io, "\\end{table}")

    return String(take!(io))
end

"""
    _build_convergence_summary_md(
        a, b, name, hsp, hxp, estp, errp,
        pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
        rule, boundary
    ) -> String

Construct a Markdown convergence-summary report.

# Function description

This helper produces a Markdown-formatted report containing:

- run configuration,
- quadrature estimates across resolutions,
- extrapolated value,
- fit results,
- explicit model expression in a math block.

The structure mirrors the [``\\LaTeX``](https://www.latex-project.org/) version for consistency.

# Arguments

Same as [`_build_convergence_summary_tex`](@ref).

# Returns

- `String`: Markdown report text.

# Notes

- Designed for GitHub-compatible Markdown rendering.
- Mathematical expressions are emitted using fenced `math` blocks.
"""
function _build_convergence_summary_md(
    a, b, name, hsp, hxp, estp, errp,
    pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
    rule, boundary
)
    io = IOBuffer()

    println(io, "# Convergence summary: $(name)")
    println(io, "")

    # ------------------------------------------------------------
    # Run configuration (column style — matches LaTeX)
    # ------------------------------------------------------------
    println(io, "## Run configuration")
    println(io, "")
    println(io, "| Interval | Rule (Boundary) | Number of error terms |")
    println(io, "|:--|:--|:--|")
    println(io,
        "| `[$(a), $(b)]` | " *
        "`$(String(rule)) ($(String(boundary)))` | " *
        "`$(nerr_terms)` |"
    )
    println(io, "")

    # ------------------------------------------------------------
    # Quadrature estimates
    # ------------------------------------------------------------
    println(io, "## Quadrature estimates and uncertainties for different step sizes")
    println(io, "")
    println(io, "| \$h\$ | \$h^$(fit_powers[2])\$ | \$I(h)\$ |")
    println(io, "|:--|:--|:--|")

    for i in eachindex(hsp)
        htxt  = _fmt_md_code_sci(hsp[i])
        hptxt = _fmt_md_code_sci(hxp[i])
        qtxt  = _fmt_avgerr_md(estp[i], errp[i])
        println(io, "| $htxt | $hptxt | $qtxt |")
    end

    htxt  = _fmt_md_code_sci(0.0)
    hptxt = _fmt_md_code_sci(0.0)
    qtxt  = _fmt_avgerr_md(I0, I0_err)

    # highlight extrapolated value (Markdown version of bold)
    println(io, "| $htxt | $hptxt | **$qtxt** |")
    println(io, "")

    # ------------------------------------------------------------
    # Fit results
    # ------------------------------------------------------------
    println(io, "## Least-chi-square fit results for extrapolation to \$h \\to 0\$")
    println(io, "")
    fit_model_tex = _build_fit_model_tex(fit_powers)
    println(io, "```math")
    println(io, fit_model_tex)
    println(io, "```")
    println(io, "")
    println(io, "| parameter | fit result |")
    println(io, "|:--|:--|")

    for i in eachindex(pvec)
        λname = fit_powers[i] == 0 ? "\$\\lambda_0\$" : "\$\\lambda_$(i-1)\$"
        λtxt  = _fmt_avgerr_md(pvec[i], λerr[i])

        # bold constant term to match LaTeX
        if fit_powers[i] == 0
            println(io, "| $λname | **$λtxt** |")
        else
            println(io, "| $λname | $λtxt |")
        end
    end

    println(io, "| \$\\chi^2 / \\text{d.o.f.}\$ | $(_fmt_md_code_sci(red)) |")

    return String(take!(io))
end