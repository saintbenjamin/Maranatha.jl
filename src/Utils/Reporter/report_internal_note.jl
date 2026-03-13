"""
    write_convergence_internal_note(
        a, b, name, hs, estimates, errors,
        fit_terms, nerr_terms, fit_result;
        rule, boundary, out_dir, save_file,
        try_build_pdf, move_existing_plots,
        author, affiliation, abstract_text
    ) -> NamedTuple

Generate a complete [``\\LaTeX``](https://www.latex-project.org/) internal-note project for a convergence study.

# Function description

This high-level routine assembles a self-contained directory containing:

- a formatted convergence-summary table,
- figure include files,
- a REVTeX-based master document,
- a Makefile for reproducible builds,
- associated plot files (optionally moved),
- optional automatic PDF compilation.

The resulting directory is suitable for archiving, sharing, or inclusion
in research workflows.

# Generated structure

```
inote_/
.tex            # master document
_table.tex      # summary tables
figs.tex        # figure includes
figs/
resultextrap.pdf
result_reldiff.pdf
Makefile
```

# Arguments

- `a`, `b`: Integration bounds.
- `name`: Identifier for the experiment or integrand.
- `hs`: Step sizes used in the convergence study.
- `estimates`: Integral estimates corresponding to `hs`.
- `errors`: Error objects or uncertainty containers.
- `fit_terms`: Number of extrapolation fit terms.
- `nerr_terms`: Number of error-model terms.
- `fit_result`: Object containing extrapolation results and metadata.

# Keyword arguments

- `rule`: Quadrature rule (default `:gauss_p3`).
- `boundary`: Boundary scheme (default `:LU_ININ`).
- `out_dir`: Output directory.
- `save_file`: If `false`, generate content without writing to disk.
- `try_build_pdf`: Attempt to build the PDF automatically.
- `move_existing_plots`: Move plot PDFs into the note directory.
- `author`: Author name for the title block.
- `affiliation`: Author affiliation.
- `abstract_text`: Optional abstract.

# Returns

A `NamedTuple` containing paths and build status information, including:

- `note_dir`
- `figs_dir`
- generated file paths
- lists of moved or missing plots
- build diagnostics
- final PDF path

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) on input inconsistencies
  or missing required fit fields.

# Notes

- Existing note directories are removed before creation.
- Plot files are expected to exist in `out_dir`.
- PDF compilation depends on external [``\\LaTeX``](https://www.latex-project.org/) tools.
"""
function write_convergence_internal_note(
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
    save_file::Bool = true,
    try_build_pdf::Bool = true,
    move_existing_plots::Bool = true,
    author::Union{Nothing,AbstractString} = nothing,
    affiliation::Union{Nothing,AbstractString} = nothing,
    abstract_text::Union{Nothing,AbstractString} = nothing,
)
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    hasproperty(fit_result, :powers) || JobLoggerTools.error_benji("fit_result missing :powers")
    hasproperty(fit_result, :params) || JobLoggerTools.error_benji("fit_result missing :params")
    hasproperty(fit_result, :cov) || JobLoggerTools.error_benji("fit_result missing :cov")
    hasproperty(fit_result, :estimate) || JobLoggerTools.error_benji("fit_result missing :estimate")
    hasproperty(fit_result, :error_estimate) || JobLoggerTools.error_benji("fit_result missing :error_estimate")

    summary_basename = _build_convergence_summary_basename(
        name,
        rule,
        boundary,
        fit_terms,
        nerr_terms,
    )

    note_dirname = "inote_" * summary_basename
    note_dir = joinpath(out_dir, note_dirname)
    figs_dir = joinpath(note_dir, "figs")

    master_basename = note_dirname
    master_tex_name = "$(master_basename).tex"
    master_pdf_name = "$(master_basename).pdf"

    if save_file
        if isdir(note_dir)
            rm(note_dir; recursive=true, force=true)
        end
        mkpath(figs_dir)
    end


    # ------------------------------------------------------------
    # 1. Write summary table tex into note directory
    # ------------------------------------------------------------
    summary_tex_name = "$(summary_basename)_table.tex"
    summary_tex_path = joinpath(note_dir, summary_tex_name)

    summary_tex = write_convergence_summary(
        a,
        b,
        name,
        hs,
        estimates,
        errors,
        fit_terms,
        nerr_terms,
        fit_result;
        rule = rule,
        boundary = boundary,
        out_dir = note_dir,
        format = :tex,
        save_file = false,
    )

    # Rename / write into the internal note naming convention
    if save_file
        open(summary_tex_path, "w") do io
            write(io, summary_tex)
        end
    end

    # ------------------------------------------------------------
    # 2. Move plot PDFs into figs/
    # ------------------------------------------------------------
    extrap_plot_name = "result_$(name)_$(String(rule))_$(String(boundary))_extrap.pdf"
    reldiff_plot_name = "result_$(name)_$(String(rule))_$(String(boundary))_reldiff.pdf"

    src_extrap = joinpath(out_dir, extrap_plot_name)
    src_reldiff = joinpath(out_dir, reldiff_plot_name)

    dst_extrap = joinpath(figs_dir, extrap_plot_name)
    dst_reldiff = joinpath(figs_dir, reldiff_plot_name)

    moved_plots = String[]
    missing_plots = String[]
    missing_plot_abort = false

    if save_file
        if !isfile(src_extrap)
            push!(missing_plots, src_extrap)
        end
        if !isfile(src_reldiff)
            push!(missing_plots, src_reldiff)
        end

        if !isempty(missing_plots)
            missing_plot_abort = true
            for fp in missing_plots
                JobLoggerTools.warn_benji("Required plot file is missing: $fp")
            end
            JobLoggerTools.warn_benji(
                "Internal note PDF build will be skipped because one or more required plot files are missing."
            )
        elseif move_existing_plots
            mv(src_extrap, dst_extrap; force=true)
            push!(moved_plots, dst_extrap)

            mv(src_reldiff, dst_reldiff; force=true)
            push!(moved_plots, dst_reldiff)
        end
    end

    # ------------------------------------------------------------
    # 3. Figure include tex
    # ------------------------------------------------------------
    fig_tex_name = "$(summary_basename)_figs.tex"
    fig_tex_path = joinpath(note_dir, fig_tex_name)

    fig_tex = _build_internal_note_figures_tex(
        extrap_plot_name,
        reldiff_plot_name;
        title = name,
        rule = rule,
        boundary = boundary,
    )

    if save_file
        open(fig_tex_path, "w") do io
            write(io, fig_tex)
        end
    end

    # ------------------------------------------------------------
    # 4. Master inote.tex
    # ------------------------------------------------------------
    master_tex_path = joinpath(note_dir, master_tex_name)
    master_tex = _build_internal_note_master_tex(
        summary_tex_name,
        fig_tex_name;
        title = name,
        rule = rule,
        boundary = boundary,
        author = author,
        affiliation = affiliation,
        abstract_text = abstract_text,
    )


    if save_file
        open(master_tex_path, "w") do io
            write(io, master_tex)
        end
    end

    # ------------------------------------------------------------
    # 5. Makefile
    # ------------------------------------------------------------
    makefile_path = joinpath(note_dir, "Makefile")
    makefile_txt = _build_internal_note_makefile(master_basename)

    if save_file
        open(makefile_path, "w") do io
            write(io, makefile_txt)
        end
    end

    # ------------------------------------------------------------
    # 6. Optional build
    # ------------------------------------------------------------
    build_attempted = false
    build_succeeded = false
    build_message = "Build not attempted."

    if save_file && try_build_pdf
        if missing_plot_abort
            build_message = "Skipped PDF build because one or more required plot files were missing."
        else
            build_attempted = true
            build_status = _try_build_internal_note(
                note_dir, 
                master_basename
            )
            build_succeeded = build_status.success
            build_message = build_status.message
        end
    end

    return (
        note_dir = note_dir,
        figs_dir = figs_dir,
        summary_tex_path = summary_tex_path,
        fig_tex_path = fig_tex_path,
        master_tex_path = master_tex_path,
        makefile_path = makefile_path,
        moved_plots = moved_plots,
        missing_plots = missing_plots,
        build_attempted = build_attempted,
        build_succeeded = build_succeeded,
        build_message = build_message,
        pdf_path = joinpath(note_dir, master_pdf_name),
    )
end

"""
    write_convergence_internal_note(result, fit_result; kwargs...) -> NamedTuple

Convenience wrapper for [`write_convergence_internal_note`](@ref).

# Function description

Extracts required data fields from a structured convergence result object
and forwards them to the primary implementation.

This allows seamless integration with pipeline outputs without manual
unpacking.

# Arguments

- `result`: Object exposing fields such as `a`, `b`, `h`, `avg`,
  `err`, `fit_terms`, `nerr_terms`, `rule`, and `boundary`.
- `fit_result`: Extrapolation fit result.

# Keyword arguments

Same as the main method.

# Returns

A `NamedTuple` identical to the primary function's return value.
"""
function write_convergence_internal_note(
    result,
    fit_result;
    name::String = "Maranatha",
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    out_dir::String = ".",
    save_file::Bool = true,
    try_build_pdf::Bool = true,
    move_existing_plots::Bool = true,
)
    return write_convergence_internal_note(
        result.a,
        result.b,
        name,
        Vector{Float64}(result.h),
        Vector{Float64}(result.avg),
        result.err,
        result.fit_terms,
        result.nerr_terms,
        fit_result;
        rule = rule,
        boundary = boundary,
        out_dir = out_dir,
        save_file = save_file,
        try_build_pdf = try_build_pdf,
        move_existing_plots = move_existing_plots,
    )
end

"""
    _build_internal_note_figures_tex(extrap_plot_name, reldiff_plot_name; title, rule, boundary) -> String

Construct a [``\\LaTeX``](https://www.latex-project.org/) figure block for inclusion in the internal note.

# Function description

Generates a two-panel figure using subfigures:

- convergence fit and extrapolation,
- relative convergence error.

Captions are formatted using `\\texorpdfstring` to ensure compatibility
with PDF bookmarks while preserving [``\\LaTeX``](https://www.latex-project.org/) formatting in the document body.

# Arguments

- `extrap_plot_name`: Filename of the extrapolation plot.
- `reldiff_plot_name`: Filename of the relative-error plot.

# Keyword arguments

- `title`: Experiment identifier.
- `rule`: Quadrature rule.
- `boundary`: Boundary scheme.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/) code for a complete figure environment.

# Notes

- Underscores are escaped for [``\\LaTeX``](https://www.latex-project.org/) safety.
- Labels are generated using the raw (unescaped) title.
"""
function _build_internal_note_figures_tex(
    extrap_plot_name::AbstractString,
    reldiff_plot_name::AbstractString;
    title,
    rule,
    boundary,
)

    # --- LaTeX-safe (for document body) ---
    safe_title = _latex_escape_underscore(title)
    safe_rule = _latex_escape_underscore(String(rule))
    safe_boundary = _latex_escape_underscore(String(boundary))

    # --- Plain text (for PDF bookmarks) ---
    plain_title = title
    plain_rule = String(rule)
    plain_boundary = String(boundary)

    # --- Caption fragments with texorpdfstring ---
    cap_title = "\\texorpdfstring{\\texttt{$safe_title}}{$plain_title}"
    cap_rule  = "\\texorpdfstring{\\texttt{$safe_rule}}{$plain_rule}"
    cap_bound = "\\texorpdfstring{\\texttt{$safe_boundary}}{$plain_boundary}"

    io = IOBuffer()

    println(io, "% Auto-generated figure include file")
    println(io, "")
    println(io, "\\begin{figure}[htbp]")
    println(io, "  \\centering")
    println(io, "  \\subfigure[Convergence fit and extrapolated result.]{")
    println(io, "    \\includegraphics[width=0.48\\textwidth]{$(extrap_plot_name)}")
    println(io, "    \\label{fig:$(title)_extrap}")
    println(io, "  }")
    println(io, "  \\hfill")
    println(io, "  \\subfigure[Relative convergence error.]{")
    println(io, "    \\includegraphics[width=0.48\\textwidth]{$(reldiff_plot_name)}")
    println(io, "    \\label{fig:$(title)_reldiff}")
    println(io, "  }")
    println(io,
        "  \\caption{Summary plots for $cap_title, with rule $cap_rule and boundary $cap_bound.}")
    println(io, "  \\label{fig:$(title)_summary}")
    println(io, "\\end{figure}")

    return String(take!(io))
end

"""
    _build_internal_note_master_tex(summary_tex_name, fig_tex_name; title, rule, boundary, author, affiliation, abstract_text) -> String

Construct the master REVTeX document for the internal note.

# Function description

Creates a complete [``\\LaTeX``](https://www.latex-project.org/) document including:

- REVTeX class configuration,
- required packages,
- title and author blocks,
- optional abstract,
- figure and summary inclusions.

The resulting document compiles independently.

# Arguments

- `summary_tex_name`: Filename of the summary table file.
- `fig_tex_name`: Filename of the figure include file.

# Keyword arguments

- `title`: Experiment identifier.
- `rule`: Quadrature rule.
- `boundary`: Boundary scheme.
- `author`: Author name (optional).
- `affiliation`: Author affiliation (optional).
- `abstract_text`: Abstract text (optional).

# Returns

- `String`: Complete [``\\LaTeX``](https://www.latex-project.org/) document source.

# Notes

- Uses `revtex4-2` with preprint-style options.
- Figure directory is assumed to be `./figs/`.
"""
function _build_internal_note_master_tex(
    summary_tex_name::AbstractString,
    fig_tex_name::AbstractString;
    title,
    rule,
    boundary,
    author::Union{Nothing,AbstractString} = nothing,
    affiliation::Union{Nothing,AbstractString} = nothing,
    abstract_text::Union{Nothing,AbstractString} = nothing,
)
    safe_title = _latex_escape_underscore(title)
    safe_rule = _latex_escape_underscore(String(rule))
    safe_boundary = _latex_escape_underscore(String(boundary))

    note_title = "Maranatha.jl: \\texttt{$safe_title}, " *
                 "\\texttt{$safe_rule}, " *
                 "\\texttt{$safe_boundary}"

    author_block = isnothing(author) ? "" : "\\author{$author}\n"
    affiliation_block = isnothing(affiliation) ? "" : "\\affiliation{$affiliation}\n"

    abstract_block = isnothing(abstract_text) ? "" : """
\\begin{abstract}
$abstract_text
\\end{abstract}
"""

    return """
%
%\\documentclass[12pt]{article}
\\documentclass[prd,onecolumn,12pt,showpacs,showkeys,preprintnumbers,floatfix,nofootinbib,superscriptaddress]{revtex4-2}

%------------------
% used packages
%------------------
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{subfigure}
\\usepackage{array}
\\usepackage{dcolumn}
\\usepackage{bm}
\\usepackage{bbold}
\\usepackage{latexsym}
\\usepackage{longtable}
\\usepackage{hyperref}
\\usepackage{mathrsfs}
\\usepackage{color}
\\usepackage{resizegather}
\\usepackage{cancel}
\\usepackage{mathtools}
\\usepackage{comment}
\\usepackage{csquotes}

\\graphicspath{{./figs/}}

%--------------------------------------------
% allow page break in the middle of equations
%--------------------------------------------
\\allowdisplaybreaks

\\begin{document}

\\title{$note_title}
$author_block$affiliation_block\\date{\\today}
$abstract_block\\maketitle

\\input{$fig_tex_name}
\\input{$summary_tex_name}

\\end{document}
"""
end

"""
    _build_internal_note_makefile(target_basename) -> String

Generate a Makefile for building the internal-note PDF.

# Function description

Produces a [``\\LaTeX``](https://www.latex-project.org/) build script supporting:

- multiple compilation passes,
- optional bibliography processing,
- figure inclusion,
- packaging and cleanup targets.

# Arguments

- `target_basename`: Base name of the document.

# Returns

- `String`: Makefile contents.

# Notes

- Uses `pdflatex` by default.
- Bibliography processing is automatically triggered when needed.
"""
function _build_internal_note_makefile(
    target_basename::AbstractString
)
    return """
#################################
TEXS := \$(wildcard *.tex)
BIBS := \$(wildcard *.bib)
BBLS := \$(wildcard *.bbl)
RTEX := \$(filter-out $(target_basename).tex,\$(TEXS))
FEPS := \$(wildcard figs/*.eps)
FPDF := \$(wildcard figs/*.pdf)

PAPERTYPE = letter
# PAPERTYPE = a4

TARGET = $(target_basename).pdf
MAIN_TEX = $(target_basename).tex

###############################
LATEX := latex
BIBTEX := bibtex
PDFLATEX := pdflatex
################################

all: \$(TARGET)

%.ps : %.dvi
\tdvips -z -t \$(PAPERTYPE) -o \$@ \$<

%.dvi : %.tex \$(RTEX) \$(BIBS) \$(FPDF)
\t\$(LATEX)  \$(basename \$<)
\t\$(BIBTEX) \$(basename \$<)
\t\$(LATEX)  \$(basename \$<)
\t\$(LATEX)  \$(basename \$<)
\t\$(LATEX)  \$(basename \$<)

%.pdf : %.tex \$(RTEX) \$(BIBS) \$(FPDF)
\t\$(PDFLATEX) \$(basename \$<)
\t@if [ -f \$(basename \$<).aux ] && grep -q "citation\\\\|bibdata\\\\|bibstyle" \$(basename \$<).aux; then \$(BIBTEX) \$(basename \$<); fi
\t\$(PDFLATEX) \$(basename \$<)
\t\$(PDFLATEX) \$(basename \$<)
\t\$(PDFLATEX) \$(basename \$<)

web:

tar:
\ttar -czvf $(target_basename).tar.gz \$(TEXS) \$(BIBS) \$(BBLS) \$(FEPS) \$(FPDF) Makefile

clean:
\trm -f \$(TARGET) *.dvi *.bbl *.aux *.end *.blg *~ *.log *.out
"""
end

"""
    _check_internal_note_latex_dependencies() -> NamedTuple

Verify availability of required [``\\LaTeX``](https://www.latex-project.org/) tools and packages.

# Function description

Performs a pre-build check for:

- `pdflatex` executable,
- optional `kpsewhich` for package discovery,
- required class and package files referenced by the preamble.

# Returns

A `NamedTuple` containing:

- `ok`: Boolean indicating whether requirements are satisfied,
- `missing`: List of missing components,
- `message`: Human-readable diagnostic.

# Notes

- If `kpsewhich` is unavailable, package checking is skipped and
  validation is deferred to the actual build process.
- Designed to fail early before expensive build attempts.
"""
function _check_internal_note_latex_dependencies()
    has_pdflatex = Sys.which("pdflatex") !== nothing
    has_kpsewhich = Sys.which("kpsewhich") !== nothing

    if !has_pdflatex
        return (
            ok = false,
            missing = String["pdflatex"],
            message = "Missing required executable: `pdflatex`.",
        )
    end

    # Files explicitly required by the current revtex-based preamble
    required_tex_files = [
        "revtex4-2.cls",
        "amsfonts.sty",
        "amssymb.sty",
        "amsmath.sty",
        "graphicx.sty",
        "subfigure.sty",
        "array.sty",
        "dcolumn.sty",
        "bm.sty",
        "bbold.sty",
        "latexsym.sty",
        "longtable.sty",
        "hyperref.sty",
        "mathrsfs.sty",
        "color.sty",
        "resizegather.sty",
        "cancel.sty",
        "mathtools.sty",
        "comment.sty",
        "csquotes.sty",
    ]

    # If kpsewhich is unavailable, we cannot pre-check package files.
    # In that case, defer the real validation to pdflatex itself.
    if !has_kpsewhich
        return (
            ok = true,
            missing = String[],
            message = "Package pre-check skipped: `kpsewhich` is unavailable. Will rely on `pdflatex` build attempt.",
        )
    end

    missing = String[]
    found = String[]

    for texfile in required_tex_files
        try
            path = readchomp(Cmd(`kpsewhich $texfile`))
            if isempty(strip(path))
                push!(missing, texfile)
            else
                push!(found, texfile)
            end
        catch
            push!(missing, texfile)
        end
    end

    if isempty(missing)
        return (
            ok = true,
            missing = String[],
            message = "All required LaTeX class/package files were found.",
        )
    else
        return (
            ok = false,
            missing = missing,
            message = "Missing required LaTeX class/package files: " * join(missing, ", "),
        )
    end
end

"""
    _try_build_internal_note(note_dir, target_basename) -> NamedTuple

Attempt to compile the internal note into a PDF.

# Function description

Build strategy:

1. Verify dependencies using `_check_internal_note_latex_dependencies`.
2. Attempt build using `make` if available.
3. Fall back to direct `pdflatex` (and `bibtex` if required).
4. Diagnose failures using log-file inspection.

# Arguments

- `note_dir`: Directory containing the [``\\LaTeX``](https://www.latex-project.org/) project.
- `target_basename`: Base name of the main document.

# Returns

A `NamedTuple` with fields:

- `success`: Build outcome,
- `message`: Detailed diagnostic or success message.

# Notes

- Multiple [``\\LaTeX``](https://www.latex-project.org/) passes are performed to resolve references.
- Bibliography processing is invoked only when requested by the `.aux` file.
- Missing TeX dependencies are extracted from the log when possible.
- The function does not throw on build failure; it reports status instead.
"""
function _try_build_internal_note(
    note_dir::AbstractString,
    target_basename::AbstractString,
)
    has_make = Sys.which("make") !== nothing
    has_pdflatex = Sys.which("pdflatex") !== nothing
    has_bibtex = Sys.which("bibtex") !== nothing

    depcheck = _check_internal_note_latex_dependencies()
    if !depcheck.ok
        return (
            success = false,
            message = depcheck.message,
        )
    end

    if !has_make && !has_pdflatex
        return (
            success = false,
            message = "Skipped PDF build: neither `make` nor `pdflatex` is available.",
        )
    end

    main_tex = "$(target_basename).tex"
    aux_path = joinpath(note_dir, "$(target_basename).aux")
    pdf_path = joinpath(note_dir, "$(target_basename).pdf")
    log_path = joinpath(note_dir, "$(target_basename).log")

    function _aux_requests_bibtex(aux_file::AbstractString)
        if !isfile(aux_file)
            return false
        end

        aux_text = read(aux_file, String)
        return occursin("citation", aux_text) ||
               occursin("bibdata", aux_text) ||
               occursin("bibstyle", aux_text)
    end

    function _extract_missing_file_from_log(log_file::AbstractString)
        if !isfile(log_file)
            return nothing
        end

        txt = read(log_file, String)

        m = match(r"! LaTeX Error: File `([^`]+)' not found\.", txt)
        if m !== nothing
            return m.captures[1]
        end

        m = match(r"! Emergency stop\.\s+<read \*> \s*([^\s]+)", txt)
        if m !== nothing
            return m.captures[1]
        end

        return nothing
    end

    function _run_pdflatex_pass()
        run(Cmd(`pdflatex -interaction=nonstopmode -halt-on-error $main_tex`; dir=note_dir))
    end

    function _run_direct_latex_build()
        try
            _run_pdflatex_pass()
        catch err
            missing_from_log = _extract_missing_file_from_log(log_path)
            if missing_from_log !== nothing
                return (
                    success = false,
                    message = "Direct LaTeX build failed: missing TeX dependency `$missing_from_log`.",
                )
            else
                rethrow(err)
            end
        end

        needs_bibtex = _aux_requests_bibtex(aux_path)

        if needs_bibtex
            if has_bibtex
                run(Cmd(`bibtex $target_basename`; dir=note_dir))
            else
                return (
                    success = false,
                    message = "PDF build requires `bibtex`, but `bibtex` is unavailable.",
                )
            end
        end

        for _ in 1:3
            try
                _run_pdflatex_pass()
            catch err
                missing_from_log = _extract_missing_file_from_log(log_path)
                if missing_from_log !== nothing
                    return (
                        success = false,
                        message = "Direct LaTeX build failed: missing TeX dependency `$missing_from_log`.",
                    )
                else
                    rethrow(err)
                end
            end
        end

        if isfile(pdf_path)
            if needs_bibtex
                return (
                    success = true,
                    message = "Successfully built PDF via direct `pdflatex`/`bibtex`: $(pdf_path)",
                )
            else
                return (
                    success = true,
                    message = "Successfully built PDF via direct `pdflatex`: $(pdf_path)",
                )
            end
        else
            return (
                success = false,
                message = "Direct LaTeX build finished, but `$(target_basename).pdf` was not found.",
            )
        end
    end

    if has_make
        if !has_bibtex
            @warn "bibtex not found. Build may still succeed if bibliography is not needed."
        end

        try
            run(Cmd(`make`; dir=note_dir))
        catch err
            if has_pdflatex
                try
                    return _run_direct_latex_build()
                catch fallback_err
                    return (
                        success = false,
                        message = "Build failed with `make`, and direct `pdflatex` fallback also failed. " *
                                  "make error: $(sprint(showerror, err)); " *
                                  "fallback error: $(sprint(showerror, fallback_err))",
                    )
                end
            else
                return (
                    success = false,
                    message = "Build failed while running `make` in $(note_dir): $(sprint(showerror, err))",
                )
            end
        end

        if isfile(pdf_path)
            return (
                success = true,
                message = "Successfully built PDF via `make`: $(pdf_path)",
            )
        else
            if has_pdflatex
                try
                    return _run_direct_latex_build()
                catch fallback_err
                    return (
                        success = false,
                        message = "Build command finished, but `$(target_basename).pdf` was not found. " *
                                  "Direct `pdflatex` fallback also failed: $(sprint(showerror, fallback_err))",
                    )
                end
            else
                return (
                    success = false,
                    message = "Build command finished, but `$(target_basename).pdf` was not found.",
                )
            end
        end
    end

    try
        return _run_direct_latex_build()
    catch err
        missing_from_log = _extract_missing_file_from_log(log_path)
        if missing_from_log !== nothing
            return (
                success = false,
                message = "Build failed while running direct `pdflatex` in $(note_dir): missing TeX dependency `$missing_from_log`.",
            )
        end

        return (
            success = false,
            message = "Build failed while running direct `pdflatex` in $(note_dir): $(sprint(showerror, err))",
        )
    end
end