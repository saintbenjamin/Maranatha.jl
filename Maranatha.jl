include("src/Maranatha.jl")
using .Maranatha

include("src/figs/PlotTools.jl")
using .PlotTools


f1d(x) = sin(x)
f2d(x, y) = sin(x) * cos(y)
f3d(x, y, z) = exp(-x^2 - y^2 - z^2)
f4d(x, y, z, t) = x * y * z * t

bounds = (0.0, π)
# ns   = [4, 8, 16, 32]
# ns_3 = [3, 6, 12, 24]

ns   = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
ns_3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]


println("▶ 1D Test Simpson 1/3")
est1, fit1, res1 = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson13)
println("Estimate: ", est1)
display(fit1)
plot_convergence_result("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13)
println()

println("▶ 1D Test Simpson 3/8")
est2, fit2, res2 = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_3, rule=:simpson38)
println("Estimate: ", est2)
display(fit2)
plot_convergence_result("1D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38)
println()

println("▶ 1D Test Bode")
est3, fit3, res3 = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:bode)
println("Estimate: ", est3)
display(fit3)
plot_convergence_result("1D", res3.h, res3.avg, res3.err, fit3; rule=:bode)
println()

# println("▶ 2D Test Simpson 1/3")
# est2, fit2, res2 = run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:simpson13)
# println("Estimate: ", est2)
# println("Fit: ", fit2)
# plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:simpson13)
# println()

# println("▶ 2D Test Simpson 3/8")
# est2, fit2, res2 = run_Maranatha(f2d, bounds...; dim=2, nsamples=ns_3, rule=:simpson38)
# println("Estimate: ", est2)
# println("Fit: ", fit2)
# plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38)
# println()

# println("▶ 2D Test Bode")
# est2, fit2, res2 = run_Maranatha(f2d, bounds...; dim=2, nsamples=ns, rule=:bode)
# println("Estimate: ", est2)
# println("Fit: ", fit2)
# plot_convergence_result("2D", res2.h, res2.avg, res2.err, fit2; rule=:bode)
# println()

# println("▶ 3D Test Simpson 1/3")
# est3, fit3, res3 = run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:simpson13)
# println("Estimate: ", est3)
# println("Fit: ", fit3)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson13)
# println()

# println("▶ 3D Test Simpson 3/8")
# est3, fit3, res3 = run_Maranatha(f3d, bounds...; dim=3, nsamples=ns_3, rule=:simpson38)
# println("Estimate: ", est3)
# println("Fit: ", fit3)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:simpson38)
# println()

# println("▶ 3D Test Bode")
# est3, fit3, res3 = run_Maranatha(f3d, bounds...; dim=3, nsamples=ns, rule=:bode)
# println("Estimate: ", est3)
# println("Fit: ", fit3)
# plot_convergence_result("3D", res3.h, res3.avg, res3.err, fit3; rule=:bode)
# println()

# println("▶ 4D Test Simpson 1/3")
# est4, fit4, res4 = run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:simpson13)
# println("Estimate: ", est4)
# println("Fit: ", fit4)
# plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson13)

# println("▶ 4D Test Simpson 3/8")
# est4, fit4, res4 = run_Maranatha(f4d, bounds...; dim=4, nsamples=ns_3, rule=:simpson38)
# println("Estimate: ", est4)
# println("Fit: ", fit4)
# plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:simpson38)

# println("▶ 4D Test Bode")
# est4, fit4, res4 = run_Maranatha(f4d, bounds...; dim=4, nsamples=ns, rule=:bode)
# println("Estimate: ", est4)
# println("Fit: ", fit4)
# plot_convergence_result("4D", res4.h, res4.avg, res4.err, fit4; rule=:bode)