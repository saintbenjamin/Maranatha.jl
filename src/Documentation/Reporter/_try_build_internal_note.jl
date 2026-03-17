# ============================================================================
# src/Documentation/Reporter/_try_build_internal_note.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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