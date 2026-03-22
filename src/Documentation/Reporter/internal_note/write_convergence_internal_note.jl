# ============================================================================
# src/Documentation/Reporter/internal_note/write_convergence_internal_note.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    write_convergence_internal_note(
        result,
        fit_result;
        name::String = "Maranatha",
        out_dir::String = ".",
        save_file::Bool = true,
        try_build_pdf::Bool = true,
        move_existing_plots::Bool = true,
        nerr_terms::Union{Nothing,Int} = nothing,
        author::Union{Nothing,AbstractString} = nothing,
        affiliation::Union{Nothing,AbstractString} = nothing,
        abstract_text::Union{Nothing,AbstractString} = nothing,
    ) -> NamedTuple

Generate a complete [``\\LaTeX``](https://www.latex-project.org/) internal-note
project for a convergence study directly from a structured result object and a
fit result.

# Function description

This high-level routine assembles a self-contained directory containing:

- a formatted convergence-summary table,
- figure include files,
- a REVTeX-based master document,
- a Makefile for reproducible builds,
- associated plot files (optionally moved),
- optional automatic PDF compilation.

It is intended for the common workflow in which the user already has a stored
or freshly computed Maranatha result object and wants to generate a complete
internal note without manually unpacking arrays.

The routine uses `result.fit_terms` together with `result.nerr_terms` or the
optional `nerr_terms` override when constructing summary/report filenames and
summary-table contents. For refinement-based error estimation, the effective
reported error-term count is forced to `0`.

# Generated structure

```
inote_<summary_basename>/
├── inote_<summary_basename>.tex
├── <summary_basename>table.tex
├── <summary_basename>figs.tex
├── figs/
│ ├── result_<name>_<spec_token>_extrap.pdf
│ └── result_<name>_<spec_token>_reldiff.pdf
└── Makefile
```

# Arguments

- `result`:
  Object exposing fields such as `h`, `avg`, `err`, `fit_terms`,
  `nerr_terms`, `rule`, `boundary`, and `err_method`.

  The note is generated from the stored result object and the companion
  `fit_result`, while the summary table is delegated to
  [`write_convergence_summary`](@ref).

- `fit_result`:
  Object containing extrapolation results and metadata.

# Keyword arguments

- `name::String = "Maranatha"`:
  Identifier used in the report title, captions, and output filenames.
- `out_dir::String = "."`:
  Output directory.
- `save_file::Bool = true`:
  If `false`, generate content without writing to disk.
- `try_build_pdf::Bool = true`:
  Attempt to build the PDF automatically.
- `move_existing_plots::Bool = true`:
  Move plot PDFs into the note directory.
- `nerr_terms::Union{Nothing,Int} = nothing`:
  Optional override for the number of error-model terms used in the report.
  If omitted, `result.nerr_terms` is used.
- `author::Union{Nothing,AbstractString} = nothing`:
  Optional author name for the REVTeX title block.
- `affiliation::Union{Nothing,AbstractString} = nothing`:
  Optional author affiliation.
- `abstract_text::Union{Nothing,AbstractString} = nothing`:
  Optional abstract text inserted into the generated note.

# Returns

A `NamedTuple` containing paths and build-status information, including:

- `note_dir`
- `figs_dir`
- `summary_tex_path`
- `fig_tex_path`
- `master_tex_path`
- `makefile_path`
- `moved_plots`
- `missing_plots`
- `build_attempted`
- `build_succeeded`
- `build_message`
- `pdf_path`

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) on input inconsistencies
  or missing required fit fields.

# Notes

- Existing note directories are removed before creation when `save_file == true`.
- Plot files are expected to exist in `out_dir`.
- PDF compilation depends on external [``\\LaTeX``](https://www.latex-project.org/)
  tools.
- This method is especially useful in notebook or pipeline workflows where the
  raw result object is already available in memory.
- For rectangular-domain workflows, the generated note is based on the
  scalarized step-size sequence `result.h` rather than the original per-axis
  step tuples.
- `<spec_token>` denotes the axis-aware rule/boundary token produced by
  [`DocUtils._rule_boundary_filename_token`](@ref).
"""
function write_convergence_internal_note(
    result,
    fit_result;
    name::String = "Maranatha",
    out_dir::String = ".",
    save_file::Bool = true,
    try_build_pdf::Bool = true,
    move_existing_plots::Bool = true,
    author::Union{Nothing,AbstractString} = nothing,
    affiliation::Union{Nothing,AbstractString} = nothing,
    abstract_text::Union{Nothing,AbstractString} = nothing,
)
    hs = Vector{Float64}(result.h)
    estimates = Vector{Float64}(result.avg)
    a = result.a
    b = result.b
    errors = result.err
    rule = result.rule
    boundary = result.boundary
    err_method = result.err_method
    fit_terms = fit_result.fit_func_terms
    nerr_terms = fit_result.nerr_terms
    nerr_terms_eff = (err_method == :refinement) ? 0 : nerr_terms

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

    summary_basename = _build_convergence_summary_basename(
        name,
        a,
        b,
        rule,
        boundary,
        fit_terms,
        nerr_terms_eff,
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
        result,
        fit_result;
        name = name,
        format = :tex,
        out_dir = note_dir,
        save_file = save_file,
    )

    if save_file
        open(summary_tex_path, "w") do io
            write(io, summary_tex)
        end
    end

    # ------------------------------------------------------------
    # 2. Move plot PDFs into figs/
    # ------------------------------------------------------------
    spec_str = DocUtils._rule_boundary_filename_token(a, b, rule, boundary)

    extrap_plot_name = "result_$(name)_$(spec_str)_extrap.pdf"
    reldiff_plot_name = "result_$(name)_$(spec_str)_reldiff.pdf"

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
