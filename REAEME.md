# Maranatha.jl: A Framework for Precise Numerical Integration

**Maranatha.jl** is a prototype numerical integration engine designed for  
flexible, multi-dimensional quadrature with error estimation and convergence extrapolation.

This framework is intended for research and educational use, focusing on  
the structure and clarity of numerical methods rather than production-level accuracy.

---

## 🔤 Name Philosophy

```
**M**eshes are generated over the interval,  
**A**daptively refined based on error estimates,  
**R**ules like Simpson’s and Bode’s are applied to evaluate,  
**A**utomatic differentiation supports higher-order error control,  
**N**umerical values of the integrand are computed at each node,  
**A**pproximation errors are analyzed and minimized,  
**T**otal values are extrapolated via least-squares fitting,  
**H**igher-dimensional extensions are made possible,  
and **A**nalysis-ready results are provided for interpretation.
```

---

## ✅ Current Features

- 1D to 4D numerical integration over hypercubes `[a, b]^d`
- Newton–Cotes rules: Simpson’s 1/3, Simpson’s 3/8, and Bode’s rule
- Automatic error estimation using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- Least-squares extrapolation as `h → 0` via [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl)
- Convergence plots with error bars and fit curves
- Modular structure: `rules/`, `error/`, `fit/`, and `Maranatha` interface

---

## ⚠️ Limitations

- Only works on **hypercube domains** with **uniform grids**
- Integration rules require strict conditions:  
  - Simpson’s 1/3: `N % 2 == 0`  
  - Simpson’s 3/8: `N % 3 == 0`  
  - Bode: `N % 4 == 0`
- Error estimates are heuristic and based on **centered local derivatives only**
- Fit extrapolation uses only the **leading-order convergence model** (`I(h) ≈ I₀ + C h^p`)
- No support yet for:
  - Adaptive mesh refinement  
  - Non-uniform quadrature nodes  
  - Singular or discontinuous integrands  
  - Variable integration bounds per dimension

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

To run a simple example:

```julia
using Maranatha

f(x) = sin(x)
I, fit, data = run_Maranatha(f, 0.0, π; dim=1, nsamples=[4,8,16,32], rule=:simpson13)
```

See the `plot_convergence_result()` function for convergence visualization.

---

## 📎 License

MIT License

---

## 🙏 Acknowledgments

This project uses:

- [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation
- [`LsqFit.jl`](https://github.com/JuliaNLSolvers/LsqFit.jl) for nonlinear least squares fitting
- [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) for visualization

---

_Maranatha_ means “Come, O Lord” — a reminder to pursue truth and beauty in every equation.