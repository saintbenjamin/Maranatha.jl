# ============================================================================
# MaranathaEntry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

include("src/Maranatha.jl")
using .Maranatha

include("src/figs/PlotTools.jl")
using .PlotTools

# # ==============================================================

# # use one fit function terms in fit_convergence()

# include("src/functions/F0000GammaEminus1.jl")
# using .F0000GammaEminus1

# # Your "f1d" integrand for Maranatha: y ↦ g(y)
# f1d(x) = gtilde_F0000(x; p=3)   # try p=2 or p=3
# f1d4(x) = gtilde_F0000(x; p=4)   # try p=2 or p=3

# ns_3 = [30, 33, 36, 39, 42, 45, 48]
# ns_4 = [40, 44, 48, 52, 56, 60, 64]

# bounds = (0.0, 1.0)

# println("▶ 1D Test Simpson 1/3 Close")
# est1, fit1, res1 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_4, rule=:simpson13_close, err_method=:derivative)
# plot_convergence_result("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_close)
# println()

# println("▶ 1D Test Simpson 3/8 Close")
# est2, fit2, res2 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_3, rule=:simpson38_close, err_method=:derivative)
# plot_convergence_result("1D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_close)
# println()

# println("▶ 1D Test Bode Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f1d4, bounds...; dim=1, nsamples=ns_4, rule=:bode_close, err_method=:derivative)
# plot_convergence_result("1D", res3.h, res3.avg, res3.err, fit3; rule=:bode_close)
# println()

# # ==============================================================

# use three fit function terms in fit_convergence()

f1d(x) = sin(x)
f2d(x, y) =  exp(-x^2 - y^2)
f3d(x, y, z) = exp(-x^2 - y^2 - z^2)
f4d(x, y, z, t) = x * y * z * t

bounds = (0.0, 1.0)
# bounds = (0.0, π)

ns   = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
ns_3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

ns_13_open   = [ 8, 12, 16, 20, 24, 28, 32, 36, 40]
ns_bode_open = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]


# println("▶ 1D Test Simpson 1/3 Close")
# est1, fit1, res1 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson13_close, err_method=:derivative)
# plot_convergence_result("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_close)
# println()

# println("▶ 1D Test Simpson 1/3 Open")
# est1, fit1, res1 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_13_open, rule=:simpson13_open, err_method=:derivative)
# plot_convergence_result("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_open)
# println()

# println("▶ 1D Test Simpson 3/8 Close")
# est2, fit2, res2 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_3, rule=:simpson38_close, err_method=:derivative)
# plot_convergence_result("1D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_close)
# println()

# println("▶ 1D Test Simpson 3/8 Open")
# est2, fit2, res2 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson38_open, err_method=:derivative)
# plot_convergence_result("1D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_open)
# println()

# println("▶ 1D Test Bode Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:bode_close, err_method=:derivative)
# plot_convergence_result("1D", res3.h, res3.avg, res3.err, fit3; rule=:bode_close)
# println()

# println("▶ 1D Test Bode Open")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_bode_open, rule=:bode_open, err_method=:derivative)
# plot_convergence_result("1D", res3.h, res3.avg, res3.err, fit3; rule=:bode_open)
# println()

# println("▶ 2D Test Simpson 1/3 Close")
# est1, fit1, res1 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:simpson13_close, err_method=:derivative)
# plot_convergence_result("2D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_close)
# println()

# println("▶ 2D Test Simpson 1/3 Open")
# est1, fit1, res1 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns_13_open, rule=:simpson13_open, err_method=:richardson)
# plot_convergence_result("2D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_open)
# println()

# println("▶ 2D Test Simpson 3/8 Close")
# est2, fit2, res2 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns_3, rule=:simpson38_close, err_method=:derivative)
# plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_close)
# println()

# println("▶ 2D Test Simpson 3/8 Open")
# est2, fit2, res2 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:simpson38_open, err_method=:derivative)
# plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_open)
# println()

# println("▶ 2D Test Bode Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:bode_close, err_method=:derivative)
# plot_convergence_result("2D", res3.h, res3.avg, res3.err, fit3; rule=:bode_close)
# println()

# println("▶ 2D Test Bode Open")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f2d, bounds...; dim=2, nsamples=ns_bode_open, rule=:bode_open, err_method=:derivative)
# plot_convergence_result("2D", res3.h, res3.avg, res3.err, fit3; rule=:bode_open)
# println()

# println("▶ 3D Test Simpson 1/3 Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:simpson13_close, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson13_close)
# println()

# println("▶ 3D Test Simpson 1/3 Open")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns_13_open, rule=:simpson13_open, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson13_open)
# println()

# println("▶ 3D Test Simpson 3/8 Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns_3, rule=:simpson38_close, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson38_close)
# println()

# println("▶ 3D Test Simpson 3/8 Open")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:simpson38_open, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson38_open)
# println()

# println("▶ 3D Test Bode Close")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:bode_close, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:bode_close)
# println()

# println("▶ 3D Test Bode Open")
# est3, fit3, res3 = Maranatha.Runner.run_Maranatha(f3d, bounds...; dim=3, nsamples=ns_bode_open, rule=:bode_open, err_method=:derivative)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:bode_open)
# println()

println("▶ 4D Test Simpson 1/3 Close")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:simpson13_close, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson13_close)

println("▶ 4D Test Simpson 1/3 Open")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns_13_open, rule=:simpson13_open, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson13_open)

println("▶ 4D Test Simpson 3/8 Close")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns_3, rule=:simpson38_close, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson38_close)

println("▶ 4D Test Simpson 3/8 Open")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:simpson38_open, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson38_open)

println("▶ 4D Test Bode Close")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:bode_close, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:bode_close)

println("▶ 4D Test Bode Open")
est4, fit4, res4 = Maranatha.Runner.run_Maranatha(f4d, bounds...; dim=4, nsamples=ns_bode_open, rule=:bode_open, err_method=:derivative)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:bode_open)