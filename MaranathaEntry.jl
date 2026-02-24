# ============================================================================
# MaranathaEntry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# using Maranatha
# using Maranatha.Z_q

# ns_3 = [30, 33, 36, 39, 42, 45, 48]
# # ns_4 = [40, 44, 48, 52, 56, 60, 64]
# # ns_4 = [80, 84, 88, 92, 96, 100, 104]

# # ns_4 = [8, 12, 16, 20, 24, 28, 32, 36, 40]
# # ns_4 = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72]
# # ns_4 = [16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64, 68, 72]
# # ns_4 = [40, 44, 48, 56, 60, 64, 68, 72, 76, 80]
# # ns_4 = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 158, 168]
# ns_4 = [12, 24, 36, 48, 60]

# ff(x1,x2,x3,x4) = integrand_Z_q((x1,x2,x3,x4))
# bounds=(0.0,π)
# rule_here=:simpson13_open

# est1, fit1, res1 = run_Maranatha(
#     ff, bounds...; dim=4, nsamples=ns_4,
#     rule=rule_here, 
#     # err_method=:derivative, 
#     err_method=:richardson, 
#     fit_terms=2
# )

# plot_convergence_result("Z_q", res1.h, res1.avg, res1.err, fit1; rule=rule_here)

using Pkg

# Usage:
#   julia --project MaranathaEntry.jl
#   MARANATHA_PLOT=1 julia --project MaranathaEntry.jl
ENV["MARANATHA_PLOT"] = get(ENV, "MARANATHA_PLOT", "0")

Pkg.test("Maranatha")