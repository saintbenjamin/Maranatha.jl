# Maranatha.jl

### Structured Tensor-Product Quadrature with Residual-Informed Extrapolation

**[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/)** 
is a research-oriented numerical quadrature framework
with a modular, rule-dispatched architecture for

* structured **multi-dimensional tensor-product integration**
* derivative-aware **residual-based error scale modeling**
* covariance-aware **least $\chi^2$ fitting for $h \to 0$ extrapolation**

It is designed for methodological research, with emphasis on 
analytical transparency, reproducibility, and modular structure.

---

## 🔗 Documentation

📘 Full documentation:

[https://saintbenjamin.github.io/Maranatha.jl/](https://saintbenjamin.github.io/Maranatha.jl/)

---

## 🧠 Core Philosophy

[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/) 
is built around a **pipeline-oriented workflow**:

1. Structured tensor-product quadrature
2. Residual-based derivative error scale modeling
3. Weighted least $\chi^2$ fitting for $h \to 0$ extrapolation
4. Covariance-propagated uncertainty visualization

Unlike traditional quadrature libraries that focus on static rule tables,
Maranatha derives rule structure through **moment / Taylor-expansion construction**
(for Newton-Cotes) and supports additional rule backends (Gauss, B-spline)
through unified tensor-product dispatch.

Fit exponents are determined from a **residual-informed expansion**
dispatched by rule family.

---

## 🚀 Current Capabilities

### 🔢 Integration

* General **multi-dimensional tensor-product quadrature** on $[a,b]^d$
* Unified quadrature dispatcher supporting:
  * Newton–Cotes (`:ns_p2`, `:ns_p3`, ...)
  * Gauss-family rules (`:gauss_p2`, `:gauss_p3`, ...)
  * B-spline-based rules (`:bsplI_p2`, .../`:bsplS_p2`, ...)
* Configurable boundary patterns (for composite rules):
  * `:LCRC`, `:LORC`, `:LCRO`, `:LORO`
* Rational composite weight assembly (for Newton–Cotes rules),
  converted to `Float64` only at the final stage

---

### 📐 Error Modeling

Residual-based derivative error *scale* models:

* Rule-family residual-term detection (midpoint-based for composite rules)
* LO / LO + NLO / multi-term support via `nerr_terms`
* Tensor-product scaling philosophy
* Automatic differentiation via:
  * [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)
  * [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/) fallback for non-finite derivatives

⚠️ These are **scaling heuristics**, not rigorous truncation bounds.

---

### 📊 Convergence Extrapolation

Weighted least $\chi^2$ fitting with:

* Residual-informed exponent basis
* Automatic power detection from rule-dispatched residual expansion
* Optional **fitting-function-shift (`ff_shift`)** to skip vanishing leading orders
* Full parameter covariance matrix
* Covariance-propagated uncertainty bands

Model form:

$$
I(h) = \sum_\texttt{i} \lambda_\texttt{i} \,  h^{\texttt{powers[i]}}
$$

where `powers` is stored in `fit_result.powers`.

---

### 📈 Visualization

* Publication-style convergence plots
* Full covariance uncertainty band
* Basis reconstruction from stored exponent vector (`fit_result.powers`)
* Convergence plotted against $h^p$, where $p$ is the leading fitted exponent
* LaTeX rendering via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)

---

### 🧩 Integrand System

Supports:

* Plain Julia functions
* Closures
* Callable structs
* Registry-based presets

Example:

```julia
struct MyIntegrand
    α::Float64
end

(f::MyIntegrand)(x) = exp(-f.α * x^2)
```

---

## 🧪 Minimal Example

Quadrature with least $\chi^2$ fitting for $h \to 0$ extrapolation:

```julia
using Maranatha

f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

I0, fit, data = run_Maranatha(
    f,
    0.0, 1.0;
    dim=4,
    nsamples=[40, 44, 48, 52, 56, 60, 64],
    rule=:ns_p5,
    boundary=:LCRC,
    err_method=:derivative,
    fit_terms=4,
    nerr_terms=2,
    ff_shift=1
)
```

Plot:

```julia
using Maranatha

plot_convergence_result(
    0.0, 1.0,
    "4D_demo",
    data.h,
    data.avg,
    data.err,
    fit;
    rule=:ns_p5,
    boundary=:LCRC
)
```

---

## ⚠️ Scope & Assumptions

* Uniform tensor-product grids only
* Hypercube domains $[a,b]^d$
* Designed for smooth integrands
* Not adaptive (yet)
* Not specialized for singular/discontinuous integrands
* High-dimensional usage scales combinatorially (intended for controlled studies)

---

## 🏗 Architecture Overview

Internal modules:

* [`Maranatha.Runner`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Runner/)
* [`Maranatha.Quadrature`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Quadrature/)
* [`Maranatha.ErrorEstimate`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/ErrorEstimate/)
* [`Maranatha.LeastChiSquareFit`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/LeastChiSquareFit/)
* [`Maranatha.PlotTools`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/PlotTools/)
* [`Maranatha.Integrands`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Integrands/)
* [`Maranatha.Utils`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Utils/)

Public API intentionally minimal:

```julia
I0, fit, data = run_Maranatha(
    f,
    0.0, 1.0;
    dim=4,
    nsamples=[40, 44, 48, 52, 56, 60, 64],
    rule=:ns_p5,
    boundary=:LCRC,
    err_method=:derivative,
    fit_terms=4,
    nerr_terms=2,
    ff_shift=1
)
plot_convergence_result(
    0.0, 1.0,
    "4D_demo",
    data.h,
    data.avg,
    data.err,
    fit;
    rule=:ns_p5,
    boundary=:LCRC
)
```

---

## 🚧 Development Status

[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/) 
is under active research development.

The architecture is stable and modular,
but high-dimensional stability and extreme quadrature regimes
are still under refinement.

---

## 📎 License

MIT License

---

## 🙏 Acknowledgments

* [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)
* [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/) 
* [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)

---

## 🔤 Name Philosophy

_[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/) as an acrostic:_

> **M**eshes define structured grids over the domain,  
> **A**cross increasing resolutions, evaluations are compared,  
> **R**ule backends enable tensor-product evaluation,  
> **A**utomatic differentiation enables derivative-based scaling,  
> **N**umerical values are computed deterministically at each node,  
> **A**pproximation errors follow residual-informed structure,  
> **T**otal values are extrapolated via weighted least chi-square fitting,  
> **H**igher-dimensional extensions preserve the tensor-product philosophy,  
> and **A**nalysis-ready results carry covariance-aware uncertainty.

With **J**u**l**ia, it is realized and goes forth into new horizons.

---

## 🔤 Name Meaning

*Maranatha* — "Come, O Lord." --- a reminder that clarity, structure, and
truth matter even in numerical computation.