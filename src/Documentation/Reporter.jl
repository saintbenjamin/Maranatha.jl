# ============================================================================
# src/Documentation/Reporter.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Reporter

Report-generation utilities for `Maranatha.jl`.

`Maranatha.Documentation.Reporter` provides routines for generating
convergence-report artifacts in LaTeX and Markdown formats.

The main public entry points are:

- [`write_convergence_summary`](@ref)
- [`write_convergence_internal_note`](@ref)
- [`write_convergence_summary_datapoints`](@ref)
- [`write_convergence_internal_note_datapoints`](@ref)

This module also provides internal helpers for numeric formatting,
LaTeX-safe text handling, report-basename construction, summary-fragment
generation, internal-note assembly, and optional PDF build checks.
"""
module Reporter

import ..LinearAlgebra
import ..Printf: @sprintf

import ..JobLoggerTools
import ..AvgErrFormatter

include("Reporter/_fmt_sci_texttt.jl")
include("Reporter/_fmt_tex_texttt_sci.jl")
include("Reporter/_fmt_md_code_sci.jl")
include("Reporter/_fmt_avgerr_tex.jl")
include("Reporter/_fmt_avgerr_md.jl")
include("Reporter/_latex_escape_underscore.jl")

include("Reporter/_build_fit_model_tex.jl")

include("Reporter/write_convergence_summary.jl")
include("Reporter/_build_convergence_summary_basename.jl")
include("Reporter/_build_convergence_summary_tex.jl")
include("Reporter/_build_convergence_summary_md.jl")

include("Reporter/write_convergence_internal_note.jl")
include("Reporter/_build_internal_note_figures_tex.jl")
include("Reporter/_build_internal_note_master_tex.jl")
include("Reporter/_build_internal_note_makefile.jl")
include("Reporter/_check_internal_note_latex_dependencies.jl")
include("Reporter/_try_build_internal_note.jl")

include("Reporter/write_convergence_summary_datapoints.jl")
include("Reporter/_build_convergence_summary_datapoints_tex.jl")
include("Reporter/_build_convergence_summary_datapoints_md.jl")
include("Reporter/_build_convergence_summary_datapoints_basename.jl")
include("Reporter/_split_report_name.jl")

include("Reporter/write_convergence_internal_note_datapoints.jl")
include("Reporter/_build_internal_note_figures_tex_datapoints.jl")

end  # module Reporter