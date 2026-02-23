# Maranatha.jl: A Framework for Precise Numerical Integration

**Maranatha.jl** is a prototype numerical integration engine designed for  
flexible, multi-dimensional quadrature with error estimation and convergence extrapolation.

This framework is intended for research and educational use, focusing on  
the structure and clarity of numerical methods rather than production-level accuracy.

---

## 🔤 Name Philosophy

**M**eshes are generated over the interval,  
**A**daptively refined based on error estimates,  
**R**ules like Simpson’s and Bode’s are applied to evaluate,  
**A**utomatic differentiation supports higher-order error control,  
**N**umerical values of the integrand are computed at each node,  
**A**pproximation errors are analyzed and minimized,  
**T**otal values are extrapolated via weighted least-squares fitting,  
**H**igher-dimensional extensions are made possible,  
and **A**nalysis-ready results are provided for interpretation.

---

## ✅ Current Features

- 1D to 4D numerical integration over hypercubes `[a, b]^d`
- Simpson’s 1/3, Simpson’s 3/8, and Bode rules  
  - closed variants
  - endpoint-free (open-chain) variants
- Automatic error estimation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- Weighted linear least-squares extrapolation as `h → 0`
- Convergence plots with error bars and fit curves
- Modular structure: `rules/`, `error/`, `fit/`, and `Maranatha` interface

---

## ⚠️ Limitations

- Currently supports **uniform tensor-product grids** on hypercube domains `[a,b]^d`.
- Both closed and endpoint-free (open-chain) rules are available, but
  grid sizes must still satisfy rule-specific structural constraints:
  - Simpson’s 1/3: `N % 2 == 0`
  - Simpson’s 3/8: `N % 3 == 0`
  - Bode: `N % 4 == 0`
- Error estimates are heuristic and based on local derivative models;
  they should be interpreted as **scale indicators**, not strict bounds.
- Convergence extrapolation assumes a leading-order power-law behavior
  (`I(h) ≈ I₀ + C h^p`) and may be unstable for poorly behaved integrands.
- No support yet for:
  - Adaptive mesh refinement
  - Non-uniform or sparse quadrature nodes
  - Integrands with strong singularities or discontinuities
  - General domains beyond tensor-product intervals

---

## 🚧 Development Status

This project is currently at the **skeleton stage**.  
Some results may be **numerically unstable or inaccurate**, especially in higher dimensions or under tight quadrature constraints.

The internal structure is designed for clarity and extensibility.  
We are actively improving stability, error modeling, and generalization capabilities.

---

## 🔜 Planned Improvements

- Adaptive grid refinement based on estimated local error
- Multi-rule switching for difficult integrands
- Symbolic error modeling and error bounding
- Integration over general domains (e.g. spheres, simplices)
- Full test coverage and benchmark suite

---

## 📂 Getting Started

A minimal 1D example:

```julia
using Maranatha
using Maranatha.PlotTools

f1d(x) = sin(x)

bounds = (0.0, π)
ns = [4, 8, 16, 32]

I, fit, data = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson13_close, err_method=:derivative)

plot_convergence_result("1D_sin", data.h, data.avg, data.err, fit; rule=:simpson13_close)
```

A more “batch-style” test setup (1D–4D) is also possible:

```julia
using Maranatha
using Maranatha.PlotTools

f1d(x) = sin(x)
f2d(x, y) = exp(-x^2 - y^2)
f3d(x, y, z) = exp(-x^2 - y^2 - z^2)
f4d(x, y, z, t) = x * y * z * t

bounds = (0.0, 1.0)

ns         = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
ns_13_open = [8, 12, 16, 20, 24, 28, 32, 36, 40]

println("▶ 1D Test Simpson 1/3 (closed)")
I, fit, data = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns, rule=:simpson13_close, err_method=:derivative)
plot_convergence_result("1D_simpson13_close", data.h, data.avg, data.err, fit; rule=:simpson13_close)

println("▶ 1D Test Simpson 1/3 (open-chain)")
I, fit, data = run_Maranatha(f1d, bounds...; dim=1, nsamples=ns_13_open, rule=:simpson13_open, err_method=:derivative)
plot_convergence_result("1D_simpson13_open", data.h, data.avg, data.err, fit; rule=:simpson13_open)
```

## 📎 License

MIT License

---

## 🙏 Acknowledgments

This project uses:

* [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation
* [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) for visualization

---

*Maranatha* means “Come, O Lord” — a reminder to pursue truth and beauty in every equation.
