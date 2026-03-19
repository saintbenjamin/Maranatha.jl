# ============================================================================
# src/Documentation/Reporter/write_convergence_internal_note_datapoints.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    write_convergence_internal_note_datapoints(
        a, b, name, hs, estimates, errors;
        h_power, xscale, yscale,
        rule, boundary, out_dir, save_file,
        try_build_pdf, move_existing_plots,
        author, affiliation, abstract_text
    ) -> NamedTuple

Generate a complete datapoints-only [``\\LaTeX``](https://www.latex-project.org/) internal-note project for a raw convergence study.

# Function description

This high-level routine assembles a self-contained internal-note directory for
raw quadrature datapoints without requiring any fitted extrapolation result.

The generated note is intended for pre-fit inspection, documentation, or
archival use when the user wants to preserve:

- the filtered quadrature datapoints,
- the plotting convention ``x = h^{p}``,
- the axis-scaling setup,
- the associated datapoints-only figure,
- a REVTeX-based buildable note structure.

Unlike [`write_convergence_internal_note`](@ref), this function does **not**
include fit parameters, extrapolated values, covariance-derived uncertainties,
or goodness-of-fit diagnostics.

# Generated structure
```
inote_<summary_basename>/
├── inote_<summary_basename>.tex
├── <summary_basename>_table.tex
├── <summary_basename>_figs.tex
├── figs/
│   └── <file_name>_<rule>_<boundary>_datapoints_*.pdf
└── Makefile
```
# Arguments

- `a`, `b`: Integration domain endpoints.

  These may be either scalars (uniform-domain case) or tuples specifying
  per-axis bounds for rectangular domains. The interval is rendered in a
  compact textual form inside the generated note.

- `name`: Identifier for the experiment, dataset, or source file.
- `hs`: Scalar step sizes used in the convergence study.

  In rectangular-domain workflows, this is expected to be the scalarized
  step-size sequence used for plotting/reporting, not the original per-axis
  step tuples.

- `estimates`: Quadrature estimates corresponding to `hs`.
- `errors`: Error objects or uncertainty containers.
  Each entry is expected to provide either:

  - a `.total` field, as in residual-based workflows, or
  - an `.estimate` field, as in refinement-based workflows.

# Keyword arguments

- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale (`:linear` or `:log`).
- `yscale`: Vertical axis scale (`:linear` or `:log`).
- `rule`: Quadrature rule label (default `:gauss_p3`).
- `boundary`: Boundary-handling label (default `:LU_ININ`).
- `out_dir`: Output directory in which the internal-note directory is created.
- `save_file`: If `false`, generate content without writing files to disk.
- `try_build_pdf`: Attempt to build the PDF automatically after file generation.
- `move_existing_plots`: Move the existing datapoints plot PDF into the note directory.
- `author`: Optional author name for the title block.
- `affiliation`: Optional author affiliation.
- `abstract_text`: Optional abstract text for the note.

# Returns

A `NamedTuple` containing generated paths and build-status information,
including:

- `note_dir`
- `figs_dir`
- generated file paths
- lists of moved or missing plots
- build diagnostics
- final PDF path

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) if input lengths are inconsistent.
- Throws if unsupported axis-scale keywords are supplied.

# Notes

- Existing note directories are removed before creation when `save_file == true`.
- The datapoints plot file is expected to already exist in `out_dir` unless
  `move_existing_plots == false`.
- The `name` argument may be a plain identifier or a path-like string; it is
  split internally into display and file-safe forms via [`_split_report_name`](@ref).
- PDF compilation depends on external [``\\LaTeX``](https://www.latex-project.org/) tools such as `pdflatex`.
- For rectangular-domain workflows, the note still reports the scalarized `hs`
  sequence supplied to this function; original per-axis step tuples are not
  embedded by this helper.
"""
function write_convergence_internal_note_datapoints(
    a,
    b,
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

    (xscale == :linear || xscale == :log) || JobLoggerTools.error_benji(
        "Unsupported xscale=$xscale (expected :linear or :log)"
    )
    (yscale == :linear || yscale == :log) || JobLoggerTools.error_benji(
        "Unsupported yscale=$yscale (expected :linear or :log)"
    )

    display_name, file_name = _split_report_name(name)

    summary_basename = _build_convergence_summary_datapoints_basename(
        file_name, rule, boundary, h_power, xscale, yscale
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

    summary_tex = write_convergence_summary_datapoints(
        a,
        b,
        display_name,
        hs,
        estimates,
        errors;
        h_power = h_power,
        xscale = xscale,
        yscale = yscale,
        rule = rule,
        boundary = boundary,
        out_dir = note_dir,
        format = :tex,
        save_file = false,
    )

    if save_file
        open(summary_tex_path, "w") do io
            write(io, summary_tex)
        end
    end

    # ------------------------------------------------------------
    # 2. Move datapoints plot PDF into figs/
    # ------------------------------------------------------------
    datapoints_plot_name =
        "$(file_name)_$(String(rule))_$(String(boundary))_datapoints_hpow_$(h_power)_$(xscale)_$(yscale).pdf"

    src_plot = joinpath(out_dir, datapoints_plot_name)
    dst_plot = joinpath(figs_dir, datapoints_plot_name)

    moved_plots = String[]
    missing_plots = String[]
    missing_plot_abort = false

    if save_file
        if !isfile(src_plot)
            push!(missing_plots, src_plot)
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
            mv(src_plot, dst_plot; force=true)
            push!(moved_plots, dst_plot)
        end
    end

    # ------------------------------------------------------------
    # 3. Figure include tex
    # ------------------------------------------------------------
    fig_tex_name = "$(summary_basename)_figs.tex"
    fig_tex_path = joinpath(note_dir, fig_tex_name)

    fig_tex = _build_internal_note_figures_tex_datapoints(
        datapoints_plot_name;
        title = display_name,
        rule = rule,
        boundary = boundary,
        h_power = h_power,
        xscale = xscale,
        yscale = yscale,
    )

    if save_file
        open(fig_tex_path, "w") do io
            write(io, fig_tex)
        end
    end

    # ------------------------------------------------------------
    # 4. Master tex
    # ------------------------------------------------------------
    master_tex_path = joinpath(note_dir, master_tex_name)
    master_tex = _build_internal_note_master_tex(
        summary_tex_name,
        fig_tex_name;
        title = display_name,
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
            build_status = _try_build_internal_note(note_dir, master_basename)
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
    write_convergence_internal_note_datapoints(
        result;
        name, h_power, xscale, yscale,
        rule, boundary, out_dir, save_file,
        try_build_pdf, move_existing_plots,
        author, affiliation, abstract_text
    ) -> NamedTuple

Convenience wrapper for [`write_convergence_internal_note_datapoints`](@ref).

# Function description

This overload extracts the required datapoint fields from a structured
quadrature result object and forwards them to the primary
[`write_convergence_internal_note_datapoints`](@ref) method.

It is intended for the common workflow in which the user already has a stored
or freshly computed Maranatha result object and wants to generate a complete
datapoints-only internal note without manually unpacking arrays.

# Arguments

- `result`: Object exposing fields such as `a`, `b`, `h`, `avg`, `err`,
  `rule`, and `boundary`.

  In rectangular-domain workflows, `result.h` is expected to be the scalarized
  step-size sequence used for plotting/reporting, while any original per-axis
  step data remain in `result.tuple_h`.

# Keyword arguments

- `name`: Identifier used in the note title, captions, and output filenames.
- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scale (`:linear` or `:log`).
- `yscale`: Vertical axis scale (`:linear` or `:log`).
- `rule`: Quadrature rule label forwarded to the primary method.
- `boundary`: Boundary-handling label forwarded to the primary method.
- `out_dir`: Output directory in which the internal-note directory is created.
- `save_file`: If `false`, generate content without writing files to disk.
- `try_build_pdf`: Attempt to build the PDF automatically after file generation.
- `move_existing_plots`: Move the existing datapoints plot PDF into the note directory.
- `author`: Optional author name for the title block.
- `affiliation`: Optional author affiliation.
- `abstract_text`: Optional abstract text for the note.

# Returns

A `NamedTuple` identical in structure to the return value of the primary
[`write_convergence_internal_note_datapoints`](@ref) method.

# Notes

- This method is especially useful in notebook or pipeline workflows where the
  raw result object is already available in memory.
- This wrapper forwards `result.h` to the primary method.
- For rectangular-domain workflows, that means the generated note is based on
  the scalarized step-size sequence rather than the original per-axis step tuples.
"""
function write_convergence_internal_note_datapoints(
    result;
    name::String = "Maranatha",
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    out_dir::String = ".",
    save_file::Bool = true,
    try_build_pdf::Bool = true,
    move_existing_plots::Bool = true,
    author::Union{Nothing,AbstractString} = nothing,
    affiliation::Union{Nothing,AbstractString} = nothing,
    abstract_text::Union{Nothing,AbstractString} = nothing,
)
    return write_convergence_internal_note_datapoints(
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
        save_file = save_file,
        try_build_pdf = try_build_pdf,
        move_existing_plots = move_existing_plots,
        author = author,
        affiliation = affiliation,
        abstract_text = abstract_text,
    )
end