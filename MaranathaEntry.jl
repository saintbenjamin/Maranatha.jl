# ============================================================================
# MaranathaEntry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

using Pkg

# Usage:
#   julia --project MaranathaEntry.jl
#   MARANATHA_PLOT=1 julia --project MaranathaEntry.jl
ENV["MARANATHA_PLOT"] = get(ENV, "MARANATHA_PLOT", "0")

Pkg.test("Maranatha")
