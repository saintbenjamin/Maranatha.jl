# ============================================================================
# src/Utils/Reporter.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Reporter

import ..LinearAlgebra
import ..Printf: @sprintf

import ..Utils.JobLoggerTools
import ..Utils.AvgErrFormatter

include("Reporter/report_formatters.jl")
include("Reporter/report_fitmodel.jl")
include("Reporter/report_summary.jl")
include("Reporter/report_internal_note.jl")
include("Reporter/report_summary_datapoints.jl")
include("Reporter/report_internal_note_datapoints.jl")

end  # module Reporter