include("src/Maranatha.jl")
using .Maranatha

include("src/figs/PlotTools.jl")
using .PlotTools


f1d(x) = sin(x)
f2d(x, y) = sin(x) * cos(y)
f3d(x, y, z) = exp(-x^2 - y^2 - z^2)
f4d(x, y, z, t) = x * y * z * t

bounds = (0.0, π)
ns = [4, 8, 16, 32]

println("▶ 1D Test")
est1, fit1, res1 = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson13)
println("Estimate: ", est1)
println("Fit: ", fit1)
plot_convergence_result("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13)
println()

println("▶ 2D Test")
est2, fit2, res2 = run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:simpson13)
println("Estimate: ", est2)
println("Fit: ", fit2)
plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:simpson13)
println()

println("▶ 3D Test")
est3, fit3, res3 = run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:simpson13)
println("Estimate: ", est3)
println("Fit: ", fit3)
plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson13)
println()

println("▶ 4D Test")
est4, fit4, res4 = run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:simpson13)
println("Estimate: ", est4)
println("Fit: ", fit4)
plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson13)