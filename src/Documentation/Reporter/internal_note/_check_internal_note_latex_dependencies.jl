# ============================================================================
# src/Documentation/Reporter/internal_note/_check_internal_note_latex_dependencies.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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