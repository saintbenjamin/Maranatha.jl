# ============================================================================
# src/Documentation/Reporter/write_convergence_internal_note.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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

In addition to forwarding the stored fit metadata from `result`, this
wrapper also allows the caller to override the fit-model settings used for
report generation via the `nterms` and `nerr_terms` keyword arguments.

# Arguments

- `result`: Object exposing fields such as `a`, `b`, `h`, `avg`,
  `err`, `fit_terms`, `nerr_terms`, `rule`, and `boundary`.
- `fit_result`: Extrapolation fit result.

# Keyword arguments

- `name::String = "Maranatha"`
  : Title or identifier used in the generated report.

- `rule::Symbol = result.rule`
  : Quadrature rule label to display in the report.

- `boundary::Symbol = result.boundary`
  : Boundary-condition label to display in the report.

- `out_dir::String = "."`
  : Output directory for generated report files.

- `save_file::Bool = true`
  : If `true`, write report files to disk.

- `try_build_pdf::Bool = true`
  : If `true`, attempt to compile the generated LaTeX source into a PDF.

- `move_existing_plots::Bool = true`
  : If `true`, move or reuse existing plot files when assembling the report.

- `nterms::Union{Nothing,Int} = nothing`
  : Optional override for the number of fit terms used in the report.
    If `nothing`, `result.fit_terms` is used.

- `nerr_terms::Union{Nothing,Int} = nothing`
  : Optional override for the number of error-model terms used in the report.
    If `nothing`, `result.nerr_terms` is used.

# Returns

A `NamedTuple` identical to the primary function's return value.

# Notes

This wrapper is intended for convenience when working directly with the
result object returned by [`Maranatha.Runner.run_Maranatha`](@ref), while
still allowing limited report-level customization without reconstructing
the full argument list manually.
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
    nterms::Union{Nothing,Int} = nothing,
    nerr_terms::Union{Nothing,Int} = nothing,
)
    fit_nterms = isnothing(nterms) ? result.fit_terms : nterms
    fit_nerr_terms = isnothing(nerr_terms) ? result.nerr_terms : nerr_terms

    return write_convergence_internal_note(
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
        save_file = save_file,
        try_build_pdf = try_build_pdf,
        move_existing_plots = move_existing_plots,
    )
end