# Maranatha.jl

**`Maranatha.jl`** is a numerical framework for **deterministic quadrature-based
continuum extrapolation** on hypercube domains ``[a,b]^n``.

Many numerical integration tools rely on stochastic sampling
(e.g. Monte Carlo or VEGAS-style algorithms).  
While such methods scale well in very high dimensions, they provide statistical
estimates whose uncertainty decreases only slowly with sampling.

`Maranatha.jl` instead focuses on **deterministic quadrature rules combined
with resolution scaling and continuum extrapolation**.

By evaluating the integral at multiple step sizes and fitting the expected
convergence behavior, the framework estimates the continuum limit
```math
h \to 0
```
together with a model-informed uncertainty.

This approach is particularly useful for **controlled numerical studies**
where convergence structure is important and deterministic sampling
can be exploited.

---

## Pipeline-oriented workflow

The design of `Maranatha.jl` follows a structured computational pipeline:

1. **Quadrature evaluation**  
2. **Error-scale estimation**
3. **continuum extrapolation using Least-``\chi^2`` fitting**
4. **Visualization of convergence behavior**

This separation allows each stage of the computation to evolve independently.

The framework exposes these responsibilities explicitly:

| Stage | Responsibility |
|:-----|:-----|
| Runner | builds raw convergence datasets |
| Quadrature | supplies tensor-product quadrature in arbitrary dimensions |
| Error estimator | supplies derivative-informed error-scale estimators |
| Fitter | performs ``h \to 0`` extrapolation |
| Plotter | visualize convergence |

---

## Architecture overview

### Integration layer

[`Maranatha.Quadrature`](@ref) provides a unified front-end for tensor-product
quadrature in arbitrary dimensions.

Supported rule families include:

- Newton–Cotes rules
- Gauss-family rules
- B-spline reconstruction rules

The quadrature core uses an **exact-moment / Taylor-expansion-based
construction** which enables a unified implementation of general
multi-point composite Newton–Cotes rules.

---

### Error modeling

[`Maranatha.ErrorEstimate`](@ref) provides lightweight **derivative-informed
error-scale estimators**.

The estimator derives scaling behavior from **rule-family residual models**
using midpoint residual moments.

Multiple residual terms can be included:

- LO
- LO + NLO
- higher-order contributions

These terms provide consistent ``h``-scaling weights used during continuum extrapolation using least-``\chi^2`` fitting.

The estimator is not a strict truncation bound; instead it produces
consistent scaling weights suitable for extrapolation.

---

### Continuum extrapolation using least ``\chi^2`` fitting

[`Maranatha.LeastChiSquareFit`](@ref) performs weighted least-``\chi^2`` fitting
to estimate the continuum limit.

The fitted model takes the form

```math
I(h) = \sum_\texttt{i} \lambda_\texttt{i} \, h^\texttt{powers[i]}
```

where the exponent basis is derived from rule-family residual models.

Optional **forward-shifting** (`ff_shift`) allows the fit to skip
nominal leading orders expected to vanish for specific integrands.

The fit routine also returns the **parameter covariance matrix**,
which enables uncertainty bands in convergence plots.

---

### Integrand system

[`Maranatha.Integrands`](@ref) implements a **registry-based preset integrand
system**.

Users may supply:

- plain Julia functions
- closures
- callable structs
- named preset integrands from the registry

This keeps the workflow flexible while enabling reproducible
benchmark problems.

---

### Execution layer

The main orchestration entry point is 

[`Maranatha.Runner.run_Maranatha`](@ref)

It performs:

1. multi-resolution quadrature
2. error-scale estimation
3. dataset construction

The resulting dataset can then be passed to

[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)

for continuum extrapolation.

---

### Plotting utilities

[`Maranatha.PlotTools`](@ref) provides visualization tools based on
[`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).

The primary plotting routines include:

- [`Maranatha.PlotTools.plot_convergence_result`](@ref)
- [`Maranatha.PlotTools.plot_quadrature_coverage_1d`](@ref)

The convergence plot displays the fitted model and its covariance-based
uncertainty band.

---

## Example workflow

The example below demonstrates a minimal end-to-end workflow
using a configuration file and a simple integrand definition.

### Step 1 :: define an integrand

First define a small integrand in a Julia souce file

Example integrand file (`sample_1d.jl`):

```julia
integrand(x) = sin(x)
```

---

### Step 2 :: prepare configuration file

Next prepare a configuration file describing the integration
domain, sampling sequence, quadrature rule, and output options.

Example configuration file (`sample_1d.toml`):

```toml
[integrand]
file = "sample_1d.jl"
name = "integrand"

[domain]
a = 0.0
b = 3.141592653589793
dim = 1

[sampling]
nsamples = [2,3,4,5,6,7,8,9]

[quadrature]
rule = "gauss_p4"
boundary = "LU_EXEX"

[error]
err_method = "forwarddiff"
fit_terms = 4
nerr_terms = 3
ff_shift = 0

[execution]
use_threads = true
```

Assume that `sample_1d.jl` and `sample_1d.toml` are located in the current working directory.

---

### Step 3 :: run quadrature pipeline

The quadrature pipeline can then be executed using the
high-level runner, producing a dataset
across multiple quadrature resolutions.

```julia
using Maranatha

run_result = run_Maranatha("sample_1d.toml")
```

---

### Step 4 :: perform continuum extrapolation

Once the dataset has been generated, the continuum limit ``h \to 0`` can be estimated by performing a least ``\chi^2`` fit.

```julia
fit_result = least_chi_square_fit(
    run_result;
    nterms = 4,
    ff_shift = 0,
    nerr_terms = 3
)

print_fit_result(fit_result)
```

---

### Step 5 :: visualize convergence

Finally, the convergence behaviour and fitted uncertainty
can be visualized using the plotting utilities.

```julia
plot_convergence_result(
    run_result, 
    fit_result;
    name = "Maranatha_test1",
    figs_dir = ".",
    save_file = true
)
```

---

## Tutorials and notebooks

For interactive demonstrations see the notebooks in

```julia
ipynb/
```

These notebooks illustrate:

- dataset generation
- merging partial runs
- filtering convergence data
- continuum extrapolation
- visualization workflows

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha,
]
Private = true
```