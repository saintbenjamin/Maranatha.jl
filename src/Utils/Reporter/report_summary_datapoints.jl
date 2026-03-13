# ============================================================================
# src/Utils/Reporter/report_summary_datapoints.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    write_convergence_summary_datapoints(
        a, b, name, hs, estimates, errors;
        h_power, xscale, yscale,
        rule, boundary, out_dir, format, save_file
    ) -> String

Generate a datapoints-only convergence summary report in [``\\LaTeX``](https://www.latex-project.org/) or Markdown format.

# Function description

This routine builds a formatted summary of raw quadrature datapoints without
requiring a fitted extrapolation model.

It is intended for pre-fit inspection or archival reporting when the user wants
to document:

- the integration interval,
- the quadrature rule and boundary configuration,
- the horizontal plotting convention ``x = h^p``,
- the filtered quadrature estimates and their associated uncertainties.

Unlike [`write_convergence_summary`](@ref), this function does **not** include
fit parameters, extrapolated values, or goodness-of-fit statistics.

# Arguments

- `a`, `b`: Integration interval endpoints.
- `name`: Identifier for the experiment, integrand, or source file.
- `hs`: Step sizes used in the quadrature study.
- `estimates`: Corresponding quadrature estimates.
- `errors`: Error objects containing pointwise uncertainties.
  Each entry is expected to provide a `.total` field in the current workflow.

# Keyword arguments

- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale (`:linear` or `:log`).
- `yscale`: Vertical axis scale (`:linear` or `:log`).
- `rule`: Quadrature rule label (default `:gauss_p3`).
- `boundary`: Boundary-handling label (default `:LU_ININ`).
- `out_dir`: Output directory used when writing the summary file.
- `format`: Output format (`:tex` or `:md`).
- `save_file`: If `true`, write the generated text to disk.

# Returns

- `String`: The generated summary text.

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) if the input lengths are inconsistent.
- Throws if unsupported axis-scale keywords are supplied.
- Throws if no valid datapoints remain after filtering.

# Notes

- Non-finite datapoints are removed automatically before reporting.
- Additional positivity filters are applied when `xscale == :log` or `yscale == :log`.
- The `name` argument may be a simple identifier or a file path; file-output
  basenames are sanitized internally via [`_split_report_name`](@ref).
- Datapoints are ordered from coarse to fine resolution (largest `h` first) in
  the final report.
"""
function write_convergence_summary_datapoints(
    a::Real,
    b::Real,
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector;
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
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

    (xscale == :linear || xscale == :log) || JobLoggerTools.error_benji(
        "Unsupported xscale=$xscale (expected :linear or :log)"
    )
    (yscale == :linear || yscale == :log) || JobLoggerTools.error_benji(
        "Unsupported yscale=$yscale (expected :linear or :log)"
    )

    display_name, file_name = _split_report_name(name)

    hxp = Float64.(hs) .^ float(h_power)
    errp = abs.([e.total for e in errors])

    mask = isfinite.(hs) .& isfinite.(hxp) .& isfinite.(estimates) .& isfinite.(errp)

    if xscale == :log
        mask .&= hxp .> 0
    end
    if yscale == :log
        mask .&= estimates .> 0
    end

    hsp  = hs[mask]
    hxp  = hxp[mask]
    estp = estimates[mask]
    errp = errp[mask]

    isempty(hsp) && JobLoggerTools.error_benji("No valid datapoints remain after filtering.")

    perm = sortperm(hsp; rev=true)
    hsp  = hsp[perm]
    hxp  = hxp[perm]
    estp = estp[perm]
    errp = errp[perm]

    summary_basename = _build_convergence_summary_datapoints_basename(
        file_name, rule, boundary, h_power, xscale, yscale
    )

    if format == :tex
        text = _build_convergence_summary_datapoints_tex(
            a, b, display_name, hsp, hxp, estp, errp;
            h_power = h_power,
            xscale = xscale,
            yscale = yscale,
            rule = rule,
            boundary = boundary,
        )
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).tex")
            open(outfile, "w") do io
                write(io, text)
            end
        end
        return text

    elseif format == :md
        text = _build_convergence_summary_datapoints_md(
            a, b, display_name, hsp, hxp, estp, errp;
            h_power = h_power,
            xscale = xscale,
            yscale = yscale,
            rule = rule,
            boundary = boundary,
        )
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).md")
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
    write_convergence_summary_datapoints(
        result;
        name, h_power, xscale, yscale,
        rule, boundary, out_dir, format, save_file
    ) -> String

Convenience wrapper for [`write_convergence_summary_datapoints`](@ref) using a result object.

# Function description

This overload extracts the required datapoint fields from a structured
quadrature result object and forwards them to the primary
[`write_convergence_summary_datapoints`](@ref) method.

It allows users to generate a datapoints-only summary directly from a stored
or freshly computed Maranatha result without manually unpacking arrays.

# Arguments

- `result`: Result object exposing fields such as `a`, `b`, `h`, `avg`,
  `err`, `rule`, and `boundary`.

# Keyword arguments

- `name`: Identifier used in the generated report and output filename.
- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale (`:linear` or `:log`).
- `yscale`: Vertical axis scale (`:linear` or `:log`).
- `rule`: Quadrature rule label forwarded to the primary method.
- `boundary`: Boundary-handling label forwarded to the primary method.
- `out_dir`: Output directory used when writing the summary file.
- `format`: Output format (`:tex` or `:md`).
- `save_file`: If `true`, write the generated text to disk.

# Returns

- `String`: The generated summary text.

# Notes

- This method is intended for the common workflow in which a user already has a
  Maranatha result object and wants to inspect or archive the raw quadrature
  datapoints before fitting.
"""
function write_convergence_summary_datapoints(
    result;
    name::String = "Maranatha",
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    out_dir::String = ".",
    format::Symbol = :tex,
    save_file::Bool = true,
)
    return write_convergence_summary_datapoints(
        result.a,
        result.b,
        name,
        Vector{Float64}(result.h),
        Vector{Float64}(result.avg),
        result.err;
        h_power = h_power,
        xscale = xscale,
        yscale = yscale,
        rule = rule,
        boundary = boundary,
        out_dir = out_dir,
        format = format,
        save_file = save_file,
    )
end

"""
    _build_convergence_summary_datapoints_basename(
        name,
        rule,
        boundary,
        h_power,
        xscale,
        yscale,
    ) -> String

Construct a standardized basename for datapoints-only convergence summary outputs.

# Function description

This helper builds a filesystem-friendly basename encoding the raw-datapoint
plotting convention used in the summary, including:

- the report or dataset name,
- the horizontal power ``h^{p}``,
- the x-axis scaling mode,
- the y-axis scaling mode.

The resulting basename is intended for use when writing summary files such as
[``\\LaTeX``](https://www.latex-project.org/) fragments or Markdown reports.

# Arguments

- `name`: User-facing identifier or file-derived label.
- `rule`: Quadrature rule label.
- `boundary`: Boundary-handling label.
- `h_power`: Power used in the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale keyword.
- `yscale`: Vertical axis scale keyword.

# Returns

- `String`: A standardized basename for datapoints-only summary artifacts.

# Notes

- The input `name` is sanitized internally via [`_split_report_name`](@ref) so
  that path-like strings or `.jld2` filenames can be used safely.
- The current basename emphasizes the datapoint-plot configuration rather than
  fit metadata.
"""
function _build_convergence_summary_datapoints_basename(
    name::AbstractString,
    rule,
    boundary,
    h_power,
    xscale,
    yscale,
)
    _, file_name = _split_report_name(name)

    return "summary_$(file_name)" *
           "_hpow_$(h_power)_$(String(xscale))_$(String(yscale))"
end

"""
    _build_convergence_summary_datapoints_tex(
        a, b, name, hsp, hxp, estp, errp;
        h_power, xscale, yscale, rule, boundary
    ) -> String

Construct a [``\\LaTeX``](https://www.latex-project.org/) datapoints-only convergence-summary fragment.

# Function description

This helper produces a [``\\LaTeX``](https://www.latex-project.org/) report fragment summarizing raw quadrature
datapoints without any fitted extrapolation model.

The generated output contains:

1. a run-configuration table,
2. a table of filtered quadrature estimates and uncertainties.

It is intended for inclusion in larger [``\\LaTeX``](https://www.latex-project.org/) documents or internal-note
workflows where the user wants to document the raw convergence data before any
fitting stage.

# Arguments

- `a`, `b`: Integration interval endpoints.
- `name`: Display name of the experiment or dataset.
- `hsp`: Filtered step sizes.
- `hxp`: Filtered transformed step sizes, typically ``h^{p}``.
- `estp`: Filtered quadrature estimates.
- `errp`: Filtered pointwise uncertainties.

# Keyword arguments

- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scaling mode.
- `yscale`: Vertical axis scaling mode.
- `rule`: Quadrature rule label.
- `boundary`: Boundary-handling label.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/)-formatted summary fragment.

# Notes

- No document preamble or `\\begin{document}` block is included.
- Numeric formatting relies on the internal helpers
  [`_fmt_tex_texttt_sci`](@ref) and [`_fmt_avgerr_tex`](@ref).
- The output is designed to parallel the style of
  [`_build_convergence_summary_tex`](@ref) while omitting fit-specific content.
"""
function _build_convergence_summary_datapoints_tex(
    a, b, name, hsp, hxp, estp, errp;
    h_power,
    xscale,
    yscale,
    rule,
    boundary,
)
    safe_name = _latex_escape_underscore(name)
    safe_rule = _latex_escape_underscore(String(rule))
    safe_boundary = _latex_escape_underscore(String(boundary))

    io = IOBuffer()

    println(io, "% Auto-generated by write_convergence_summary_datapoints")
    println(io, "")

    println(io, "\\begin{table}[ht!]")
    println(io, "\\begin{ruledtabular}")
    println(io, "\\caption{Run configuration for \\texttt{$(safe_name)}}")
    println(io, "\\begin{tabular}{ @{\\quad} c @{\\quad} | @{\\quad} c @{\\quad} | @{\\quad} c @{\\quad} }")
    println(io, "Interval & Rule (Boundary) & Plot setup \\\\")
    println(io, "\\hline")
    println(io,
        "\$[$(a), $(b)]\$ & " *
        "\\texttt{$(safe_rule)} (\\texttt{$(safe_boundary)}) & " *
        "\$h^{$(h_power)}\$, \\texttt{$(String(xscale))}/\\texttt{$(String(yscale))} \\\\"
    )
    println(io, "\\end{tabular}")
    println(io, "\\end{ruledtabular}")
    println(io, "\\end{table}")
    println(io, "")

    println(io, "\\begin{table}[ht!]")
    println(io, "\\begin{ruledtabular}")
    println(io, "\\caption{Quadrature estimates and uncertainties for different step sizes}")
    println(io, "\\begin{tabular}{@{\\quad} l @{\\quad} l @{\\quad} | @{\\quad} l @{\\quad}}")
    println(io, "\$h\$ & \$h^{$(h_power)}\$ & \$\\hphantom{-}I(h)\$ \\\\")
    println(io, "\\hline")

    for i in eachindex(hsp)
        htxt  = _fmt_tex_texttt_sci(hsp[i])
        hptxt = _fmt_tex_texttt_sci(hxp[i])
        qtxt  = _fmt_avgerr_tex(estp[i], errp[i])
        qtxt  = startswith(qtxt, "-") ? qtxt : "\\hphantom{-}$qtxt"
        println(io, "$htxt & $hptxt & \$$qtxt\$ \\\\")
    end

    println(io, "\\end{tabular}")
    println(io, "\\end{ruledtabular}")
    println(io, "\\end{table}")

    return String(take!(io))
end

"""
    _build_convergence_summary_datapoints_md(
        a, b, name, hsp, hxp, estp, errp;
        h_power, xscale, yscale, rule, boundary
    ) -> String

Construct a Markdown datapoints-only convergence-summary report.

# Function description

This helper builds a Markdown representation of raw quadrature datapoints for
inspection, documentation, or lightweight archival use.

The generated report contains:

- a run-configuration section,
- a table of filtered step sizes,
- transformed horizontal coordinates,
- quadrature estimates with uncertainties.

Unlike the fit-based reporting helpers, this routine focuses exclusively on the
measured datapoints and plotting convention, without including any extrapolated
value or fit diagnostics.

# Arguments

- `a`, `b`: Integration interval endpoints.
- `name`: Display name of the experiment or dataset.
- `hsp`: Filtered step sizes.
- `hxp`: Filtered transformed step sizes, typically ``h^{p}``.
- `estp`: Filtered quadrature estimates.
- `errp`: Filtered pointwise uncertainties.

# Keyword arguments

- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scaling mode.
- `yscale`: Vertical axis scaling mode.
- `rule`: Quadrature rule label.
- `boundary`: Boundary-handling label.

# Returns

- `String`: Markdown-formatted report text.

# Notes

- Numeric formatting relies on the internal helpers
  [`_fmt_md_code_sci`](@ref) and [`_fmt_avgerr_md`](@ref).
- The output is intended to be GitHub-friendly and visually parallel to the
  [``\\LaTeX``](https://www.latex-project.org/) version where possible.
"""
function _build_convergence_summary_datapoints_md(
    a, b, name, hsp, hxp, estp, errp;
    h_power,
    xscale,
    yscale,
    rule,
    boundary,
)
    io = IOBuffer()

    println(io, "# Convergence datapoints summary: $(name)")
    println(io, "")

    println(io, "## Run configuration")
    println(io, "")
    println(io, "| Interval | Rule (Boundary) | Plot setup |")
    println(io, "|:--|:--|:--|")
    println(io,
        "| `[$(a), $(b)]` | " *
        "`$(String(rule)) ($(String(boundary)))` | " *
        "`h^$(h_power)`, `$(String(xscale))/$(String(yscale))` |"
    )
    println(io, "")

    println(io, "## Quadrature estimates and uncertainties for different step sizes")
    println(io, "")
    println(io, "| \$h\$ | \$h^$(h_power)\$ | \$I(h)\$ |")
    println(io, "|:--|:--|:--|")

    for i in eachindex(hsp)
        htxt  = _fmt_md_code_sci(hsp[i])
        hptxt = _fmt_md_code_sci(hxp[i])
        qtxt  = _fmt_avgerr_md(estp[i], errp[i])
        println(io, "| $htxt | $hptxt | $qtxt |")
    end

    return String(take!(io))
end

"""
    _split_report_name(name::AbstractString) -> Tuple{String,String}

Split a user-supplied report name into display and file-safe components.

# Function description

This helper separates a report identifier into two related forms:

1. `display_name`: the original string, preserved for human-readable use in
   report titles or captions,
2. `file_name`: a sanitized basename suitable for filesystem output.

It is especially useful when users pass a full file path, such as a stored
`.jld2` result file, as the `name` argument to reporting functions.

# Arguments

- `name::AbstractString`: Original report name, identifier, or file path.

# Returns

- `Tuple{String,String}`:
  A pair `(display_name, file_name)` where:
  - `display_name` preserves the original string,
  - `file_name` is reduced to `basename(name)` with a trailing `.jld2`
    suffix removed when present.

# Notes

- This helper prevents accidental insertion of directory separators into
  generated output filenames.
- It is intended for internal use by reporting and plotting helpers that need
  to distinguish between display labels and filesystem-safe basenames.
"""
function _split_report_name(name::AbstractString)
    display_name = String(name)
    file_name = replace(basename(String(name)), r"\.jld2$" => "")
    return display_name, file_name
end