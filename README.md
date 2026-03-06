# Maranatha.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18890716.svg)](https://doi.org/10.5281/zenodo.18890716)

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

## 📦 Installation

Install from the Julia package registry:

```julia
using Pkg
Pkg.add("Maranatha")
```

---

## ⚡ Quick Start

A typical workflow in `Maranatha.jl` is:

```julia
using Maranatha

f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

print_fit_result(fit)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```

This three-step pipeline reflects the core design of the package:

1. generate a convergence dataset,
2. fit the extrapolation model,
3. inspect and visualize the result.

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

Fit exponents used in the extrapolation model are determined from a
**rule-dispatched residual expansion** that detects the leading
non-vanishing truncation orders for each quadrature family.

---

## 🎯 Intended Use

`Maranatha.jl` is designed for **controlled numerical studies** rather than
drop-in black-box integration.

It is especially suitable when you want to:

* compare convergence behavior across quadrature families,
* study residual-informed extrapolation toward $h \to 0$,
* inspect error-scaling structure in a reproducible tensor-product setting.

It is **not** intended as a general-purpose adaptive integrator for arbitrary
non-smooth or singular problems.

---

## 🚀 Current Capabilities

### 🔢 Integration

* General **multi-dimensional tensor-product quadrature** on $[a,b]^d$
* Unified quadrature dispatcher supporting:
  * Newton–Cotes (`:newton_p2`, `:newton_p3`, ...)
  * Gauss-family rules (`:gauss_p2`, `:gauss_p3`, ...)
  * B-spline-based rules (`:bspline_interp_p2`, .../`:bspline_smooth_p2`, ...)
* Configurable boundary patterns (for composite rules):
  * `:LU_ININ`, `:LU_EXIN`, `:LU_INEX`, `:LU_EXEX`
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
  * [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/)
  * [`FastDifferentiation.jl`](https://brianguenter.github.io/FastDifferentiation.jl/stable/)
  * [`Enzyme.jl`](https://enzyme.mit.edu/julia/stable/)

⚠️ These models provide **error scaling estimates**, not strict
truncation bounds. Their purpose is to stabilize weighted
$\chi^2$ extrapolation rather than provide conservative error bounds.

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

where `powers` is determined automatically from the rule-dispatched
residual expansion and stored in `fit_result.powers`.

---

### 📈 Visualization

* Publication-style convergence plots
* Full covariance uncertainty band
* Basis reconstruction from stored exponent vector (`fit_result.powers`)
* Convergence plotted against $h^p$, where $p$ is the first non-constant
  exponent stored in `fit_result.powers`
* LaTeX rendering via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)

---

### 🧩 Integrand System

Supports:

* Plain Julia functions
* Closures
* Callable structs
* Registry-based presets

Integrands must accept `dim` scalar arguments corresponding to the
tensor-product quadrature dimension.

Example:

```julia
struct MyIntegrand
    α::Float64
end

(f::MyIntegrand)(x) = exp(-f.α * x^2)
```

---

## 🧪 Minimal Example

The example below demonstrates the standard three-step workflow:
generate a convergence dataset, perform the downstream fit, and plot the resulting convergence curve.

```julia
f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```

---

## 📦 Returned Objects

### `result = run_Maranatha(...)`

The runner returns a `NamedTuple` containing the raw convergence-study data, including:

* `result.a`, `result.b` — integration bounds
* `result.h` — step sizes
* `result.avg` — quadrature estimates
* `result.err` — error-information objects
* `result.rule`, `result.boundary` — quadrature configuration
* `result.fit_terms`, `result.ff_shift`, `result.nerr_terms` — downstream fitting metadata

### `fit = least_chi_square_fit(...)`

The fitter returns a `NamedTuple` containing extrapolation results, including:

* `fit.estimate` — extrapolated value $I(h \to 0)$
* `fit.error_estimate` — uncertainty of the extrapolated value
* `fit.params` — fitted parameter vector
* `fit.param_errors` — parameter uncertainties
* `fit.cov` — covariance matrix
* `fit.powers` — exponent basis used by the fit
* `fit.chisq`, `fit.redchisq`, `fit.dof` — fit diagnostics

---

## ⚠️ Scope & Assumptions

* Uniform tensor-product grids only
* Hypercube domains $[a,b]^d$
* Designed primarily for smooth integrands
* Not adaptive (yet)
* Not specialized for singular or discontinuous integrands
* High-dimensional usage scales combinatorially (therefore intended mainly for controlled studies)

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

The public API is intentionally minimal and centered around
a pipeline-style workflow:

```julia
result = run_Maranatha(...)
fit = least_chi_square_fit(...)
plot_convergence_result(...)
```

---

## 🚧 Development Status

[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/) 
is under active research development.

The core architecture is stable and largely settled.
Ongoing development focuses on:

* robustness of high-dimensional extrapolation
* improved stability of error-scale modeling
* additional quadrature rule families

---

## 📚 Citation

If you use **Maranatha.jl** in research, please cite:

```bibtex
@software{Choi:2026maranatha,
  author       = {Benjamin J. Choi},
  title        = {\href{https://doi.org/10.5281/zenodo.18890716}{Maranatha.jl: Numerical Quadrature Continuum Extrapolation Framework in Julia}},
  month        = mar,
  year         = 2026,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18890716}
}
```

---

## 📎 License

MIT License

---

## 🙏 Acknowledgments

* [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)
* [`TaylorSeries.jl`](https://juliadiff.org/TaylorSeries.jl/stable/)
* [`FastDifferentiation.jl`](https://brianguenter.github.io/FastDifferentiation.jl/stable/)
* [`Enzyme.jl`](https://enzyme.mit.edu/julia/stable/)
* [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)

---

## 🔤 Name Philosophy

_[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/) as an acrostic:_

> **M**eshes sketch structured grids over the domain,  
> **A**cross increasing resolutions, evaluations gather and compare.  
> **R**ule backends quietly orchestrate tensor-product evaluation,  
> **A**utomatic differentiation keeps an eye on derivative-based scaling.  
> **N**umerical values take their places at each node,  
> **A**pproximation errors reveal their residual-informed structure.  
> **T**otal values emerge through weighted least chi-square fitting,  
> **H**igher-dimensional extensions remain faithful to the tensor-product philosophy,  
> and **A**nalysis-ready results carry covariance-aware uncertainty without apology.  

With **J**u**l**ia, it is realized and goes forth into new horizons.

---

## 🔤 Name Meaning

*Maranatha* — "Come, O Lord." --- a reminder that clarity, structure, and
truth matter even in numerical computation.
