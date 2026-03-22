# ============================================================================
# src/Documentation/Reporter/Reporter.jl
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
import ..Documentation.DocUtils

include("format/_fmt_sci_texttt.jl")
include("format/_fmt_tex_texttt_sci.jl")
include("format/_fmt_md_code_sci.jl")
include("format/_fmt_avgerr_tex.jl")
include("format/_fmt_avgerr_md.jl")
include("format/_latex_escape_underscore.jl")
include("format/_fmt_axis_interval_for_run_config.jl")
include("format/_fmt_axis_cell_md.jl")
include("format/_fmt_axis_cell_tex.jl")
include("format/_fmt_rule_boundary_cell_md.jl")
include("format/_fmt_rule_boundary_cell_tex.jl")
include("format/_build_fit_model_tex.jl")

include("run_config/_report_cfg_is_multi.jl")
include("run_config/_report_cfg_dim.jl")
include("run_config/_report_cfg_at.jl")

include("summary/write_convergence_summary.jl")
include("summary/_build_convergence_summary_basename.jl")
include("summary/_build_convergence_summary_tex.jl")
include("summary/_build_convergence_summary_md.jl")

include("internal_note/write_convergence_internal_note.jl")
include("internal_note/_build_internal_note_figures_tex.jl")
include("internal_note/_build_internal_note_master_tex.jl")
include("internal_note/_build_internal_note_makefile.jl")
include("internal_note/_check_internal_note_latex_dependencies.jl")
include("internal_note/_try_build_internal_note.jl")

include("datapoints/write_convergence_summary_datapoints.jl")
include("datapoints/_build_convergence_summary_datapoints_tex.jl")
include("datapoints/_build_convergence_summary_datapoints_md.jl")
include("datapoints/_build_convergence_summary_datapoints_basename.jl")
include("datapoints/write_convergence_internal_note_datapoints.jl")
include("datapoints/_build_internal_note_figures_tex_datapoints.jl")

end  # module Reporter
