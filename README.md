# Maranatha.jl

### Structured Newton–Cotes Quadrature with Residual-Informed Extrapolation

**Maranatha.jl** is a research-oriented numerical quadrature framework for

* structured **multi-dimensional tensor-product integration**
* derivative-aware **residual-based error scale modeling**
* covariance-aware **least-χ² extrapolation to `h → 0`**

It is designed for methodological research and lattice perturbation theory experiments,
with emphasis on analytical transparency, reproducibility, and modular structure.

---

## 🔗 Documentation

📘 Full documentation:

[https://saintbenjamin.github.io/Maranatha.jl/](https://saintbenjamin.github.io/Maranatha.jl/)

---

## 🧠 Core Philosophy

Maranatha is built around a **pipeline-oriented workflow**:

1. Structured tensor-product quadrature
2. Residual-based derivative error scale modeling
3. Weighted least-χ² extrapolation (`h → 0`)
4. Covariance-propagated uncertainty visualization

Unlike traditional quadrature libraries that focus on rule tables,
Maranatha derives rule structure through **moment / Taylor-expansion construction**,
and derives fit exponents from the **composite midpoint residual expansion**.

---

## 🚀 Current Capabilities

### 🔢 Integration

* General **multi-dimensional tensor-product quadrature** on `[a,b]^d`
* Unified `:ns_pK` Newton–Cotes generator (no legacy rule tables)
* Configurable boundary patterns:

  * `:LCRC`, `:LORC`, `:LCRO`, `:LORO`
* Rational composite weight assembly (converted to Float64 only at final stage)

---

### 📐 Error Modeling

Residual-based derivative error *scale* models:

* Midpoint residual-moment detection
* LO / LO+NLO / multi-term support via `nerr_terms`
* Tensor-product scaling philosophy
* Automatic differentiation via:

  * [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/)
  * [TaylorSeries.jl](https://juliadiff.org/TaylorSeries.jl/stable/) fallback for non-finite derivatives

⚠️ These are **scaling heuristics**, not rigorous truncation bounds.

---

### 📊 Convergence Extrapolation

Weighted least-χ² fitting with:

* Residual-informed exponent basis
* Automatic power detection from composite midpoint expansion
* Optional **fitting-function-shift (`ff_shift`)** to skip vanishing leading orders
* Full parameter covariance matrix
* Covariance-propagated uncertainty bands

Model form:

```
I(h) = Σ λᵢ h^{powers[i]}
```

where `powers` is stored in `fit_result.powers`.

---

### 📈 Visualization

* Publication-style convergence plots
* Full covariance uncertainty band
* Basis reconstruction from stored exponent vector
* LaTeX rendering via [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)

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
    0.0, 1,
    "4D_demo",
    data.h,
    data.avg,
    data.err,
    fit;
    rule=:ns_p3,
    boundary=:LCRC
)
```

---

## ⚠️ Scope & Assumptions

* Uniform tensor-product grids only
* Hypercube domains `[a,b]^d`
* Smooth integrands preferred
* Not adaptive (yet)
* Not designed for singular/discontinuous integrands

---

## 🏗 Architecture Overview

Internal modules:

* `Integrate`
* `ErrorEstimator`
* `LeastChiSquareFit`
* `Integrands`
* `Runner`
* `PlotTools`
* `JobLoggerTools`

Public API intentionally minimal:

```julia
run_Maranatha
plot_convergence_result
```

---

## 🚧 Development Status

`Maranatha.jl` is under active research development.

The architecture is stable and modular,
but high-dimensional stability and extreme quadrature regimes
are still under refinement.

---

## 📎 License

MIT License

---

## 🙏 Acknowledgments

* [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/)
* [TaylorSeries.jl](https://juliadiff.org/TaylorSeries.jl/stable/) 
* [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)

---

## 🔤 Name Meaning

*Maranatha* — "Come, O Lord." --- a reminder that clarity, structure, and
truth matter even in numerical computation.