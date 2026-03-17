# ============================================================================
# src/Documentation/Reporter/_build_internal_note_makefile.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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