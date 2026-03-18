# Maranatha.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18890716.svg)](https://doi.org/10.5281/zenodo.18890716)

### A Deterministic Quadrature Framework for Convergence Analysis and Continuum Extrapolation

**[`Maranatha.jl`](https://saintbenjamin.github.io/Maranatha.jl/)** 
is a research-oriented numerical quadrature framework
with a modular, rule-dispatched architecture for

* structured **multi-dimensional tensor-product integration**
* derivative-aware **residual-based error scale modeling**
* covariance-aware **least $\chi^2$ fitting for $h \to 0$ extrapolation**
* optional **threaded-subgrid CPU execution** and **CUDA-based GPU execution**

It is designed for methodological research, with emphasis on
analytical transparency, reproducibility, modular structure,
and explicit control over execution backends.

---

## 📦 Installation

Install from the Julia package registry:

```julia
using Pkg
Pkg.add("Maranatha")
```

---

## ⚡ Quick Start

The example below demonstrates a minimal end-to-end workflow
using a configuration file and a simple integrand definition.

First define a small integrand in a Julia source file.

Example integrand (`sample_1d.jl`)

```julia
integrand(x) = sin(x)
```

Next prepare a configuration file describing the integration
domain, sampling sequence, quadrature rule, and output options.

Configuration file (`sample_1d.toml`)

```toml
[integrand]
file = "sample_1d.jl"
name = "integrand"

[domain]
a = 0.0
b = 3.141592653589793
dim = 1

[sampling]
nsamples = [2, 3, 4, 5, 6, 7, 8, 9]

[quadrature]
rule = "gauss_p4"
boundary = "LU_EXEX"

[error]
err_method = "refinement"
fit_terms = 4
nerr_terms = 3
ff_shift = 0

[execution]
use_error_jet = true

[output]
name_prefix = "1D"
save_path = "."
write_summary = true
save_file = true
```

Assume that `sample_1d.jl` and `sample_1d.toml` are located
in the current working directory.

Execution backend selection depends on runtime settings:

- serial execution is used by default,
- threaded subgrid execution is used when Julia is started with multiple CPU threads,
- CUDA execution is used only when `use_cuda = true` is explicitly requested.

For example, CPU threading can be enabled before starting Julia with:

```bash
JULIA_NUM_THREADS=8 julia
```

The quadrature pipeline can then be executed using the
high-level runner, producing a convergence dataset 
across multiple quadrature resolutions.

```julia
using Maranatha

run_result = run_Maranatha("./sample_1d.toml")
```

To explicitly request GPU execution, use the direct-call interface and set
`use_cuda = true`:

```julia
run_result = run_Maranatha(
    integrand,
    0.0,
    pi;
    dim = 1,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :refinement,
    use_cuda = true,
)
```

Once the dataset has been generated, the continuum limit
$h \to 0$ can be estimated by performing a least $\chi^2$ fit.

```julia
fit_result = least_chi_square_fit(
    run_result; 
    nterms=3, 
    ff_shift=0, 
    nerr_terms=2
)

print_fit_result(fit_result)
```

Finally, the convergence behavior and fitted uncertainty
can be visualized using the plotting utilities.

```julia
plot_convergence_result(
    run_result, 
    fit_result;
    name="Maranatha_test1",
    figs_dir=".",
    save_file=true
)
```

For more detailed examples and interactive demonstrations,
see the tutorial notebooks in the `ipynb/` directory of this repository.

These notebooks provide step-by-step tutorials covering the full
`Maranatha.jl` workflow, including dataset generation, merging partial
runs, filtering datapoints, and convergence visualization.

---

### ⚡ Why runs can be fast

When the refinement-based error model is used, Maranatha avoids computing
high-order derivatives of the integrand entirely. Instead, it infers
error scaling directly from resolution refinement.

This makes the framework practical even for computationally expensive
integrands, where derivative-based error estimation would dominate the
runtime.

In many cases, refinement-based estimates provide error scales close to
those obtained from theoretical derivative models while remaining much
faster to compute.

When combined with threaded subgrid execution on multi-core CPUs, this can
substantially reduce wall-clock time for large tensor-product workloads.

---

## 🔗 Documentation

📘 Documentation:

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

Fit exponents used in the extrapolation model are determined from a
**rule-dispatched residual expansion** that detects the leading
non-vanishing truncation orders for each quadrature family.

A key design choice is the availability of refinement-based error
estimation, which prioritizes observable convergence behavior over
analytic derivative information. This reflects the framework's emphasis
on practical numerical studies rather than purely theoretical error
bounds.

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
* Multiple execution backends:
  * serial tensor-product evaluation
  * threaded subgrid CPU backend
  * CUDA-based GPU backend

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

### 🚀 Refinement-based error estimation

In addition to derivative-based backends, Maranatha supports a
**refinement-based error scale model** that estimates truncation behavior
directly from differences between successive quadrature resolutions.

This approach has several practical advantages:

- **No high-order derivatives are required**
- Avoids expensive automatic differentiation passes
- Scales efficiently to complex integrands and higher dimensions
- Typically much faster than derivative-based estimators

Extensive testing indicates that the resulting error scales are often
comparable in quality to fully derivative-based models for smooth
integrands, while being significantly cheaper to evaluate.

Because it relies only on observable convergence behavior,
the refinement method is particularly robust for complicated
integrands where high-order derivatives are costly or unstable
to compute.

This also makes the refinement path especially compatible with accelerated
execution backends such as threaded subgrid CPU evaluation and CUDA-based
quadrature.

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
* [$\LaTeX$](https://www.latex-project.org/) rendering via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)

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
* CUDA execution requires a compatible CUDA environment and GPU-callable integrands

---

## 🏗 Architecture Overview

Internal modules:

* [`Maranatha.Runner`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Runner/)
* [`Maranatha.Quadrature`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Quadrature/)
* [`Maranatha.ErrorEstimate`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/ErrorEstimate/)
* [`Maranatha.LeastChiSquareFit`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/LeastChiSquareFit/)
* [`Maranatha.Documentation`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Documentation/)
* [`Maranatha.Integrands`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Integrands/)
* [`Maranatha.Utils`](https://saintbenjamin.github.io/Maranatha.jl/stable/lib/Utils/) — shared infrastructure, including reporting and reproducibility tools

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
* continued optimization of CPU-threaded and CUDA execution backends

---

## 🧑‍🔬 Citation

If you use **Maranatha.jl** in research, please cite:

```bibtex
@misc{Choi:2026maranatha,
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
* [`CUDA.jl`](https://cuda.juliagpu.org/stable/)

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
