# ============================================================================
# src/Utils/AvgErrFormatter.jl (Benji: taken from src/Sarah/AvgErrFormatter.jl of Deborah.jl)
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Utils

include("JobLoggerTools.jl")
include("AvgErrFormatter.jl")

using .JobLoggerTools
using .AvgErrFormatter

end  # module Utils