# Maranatha.jl

**`Maranatha.jl`** is a numerical framework for **deterministic quadrature-based
continuum extrapolation** on hyperrectangular domains

```math
\prod_{i=1}^n [a_i, b_i]
```

including the special case of hypercubes ``[a,b]^n``.

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

The integration bounds may be specified either as common scalar limits
for all axes or as axis-wise endpoints, allowing general rectangular domains.

---

## Pipeline-oriented workflow

The design of `Maranatha.jl` follows a structured computational pipeline:

1. **Quadrature evaluation**  
2. **Error-scale estimation**
3. **Continuum extrapolation using Least-``\chi^2`` fitting**
4. **Structured reporting and archival output**
5. **Visualization of convergence behavior**

This separation allows each stage of the computation and post-processing
workflow to evolve independently.

The framework exposes these responsibilities explicitly:

| Stage | Responsibility |
|:-----|:-----|
| Runner | builds raw convergence datasets |
| Quadrature | supplies tensor-product quadrature in arbitrary dimensions |
| Error estimator | supplies refinement-based or derivative-based error-scale estimators |
| Fitter | performs ``h \to 0`` extrapolation |
| Plotter | visualizes convergence behavior |
| Reporter | generates structured report artifacts and internal-note outputs |

---

## Architecture overview

### Integration layer

[`Maranatha.Quadrature`](@ref) provides a unified front-end for tensor-product
quadrature in arbitrary dimensions on hyperrectangular domains.

Supported rule families include:

- Newton–Cotes rules
- Gauss-family rules
- B-spline reconstruction rules

The quadrature core uses an **exact-moment / Taylor-expansion-based
construction** which enables a unified implementation of general
multi-point composite Newton–Cotes rules.

The quadrature layer supports multiple execution backends:

- **Serial evaluation** (default)
- **Threaded subgrid partitioning** for multi-core CPUs
- **CUDA-based GPU acceleration** for massively parallel workloads

The backend is selected automatically based on configuration
options and hardware availability.

---

### Error modeling

[`Maranatha.ErrorEstimate`](@ref) provides lightweight **error-scale estimators**
used to characterize the resolution dependence of quadrature results.

Two complementary approaches are supported:

- **Refinement-based estimation**  
  compares results at different resolutions (e.g. `h` vs. `h/2`) and measures
  how the computed integral changes under grid refinement.

- **Derivative-based estimation**  
  derives scaling behavior from rule-family residual models using midpoint
  residual moments and derivative probes of the integrand.

For derivative-based estimators, multiple residual terms can be included:

- LO
- LO + NLO
- higher-order contributions

These terms provide consistent ``h``-scaling weights used during continuum
extrapolation via least-``\chi^2`` fitting.

The estimator is not a strict truncation bound; instead it produces
consistent scaling weights suitable for extrapolation.

---

### Continuum extrapolation using least-``\chi^2`` fitting

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
[`Maranatha.Runner.run_Maranatha`](@ref).

The integration domain supplied to the runner may be a scalar interval
(applied to all axes) or axis-specific limits given as tuples or vectors.

It performs:

1. multi-resolution quadrature
2. error-scale estimation (refinement-based or derivative-based)
3. dataset construction

The resulting dataset can then be passed to

[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)

for continuum extrapolation.

---

### Plotting utilities

[`Maranatha.Documentation.PlotTools`](@ref) provides visualization tools based on
[`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).

The primary plotting routines include:

- [`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref)
- [`Maranatha.Documentation.PlotTools.plot_quadrature_coverage_1d`](@ref)

The convergence plot displays the fitted model and its covariance-based
uncertainty band.

---

### Reporting utilities

High-level report generation is provided by
[`Maranatha.Documentation.Reporter`](@ref).

This layer is responsible for turning numerical and plotting results into
structured, shareable documentation artifacts such as summary tables,
``\LaTeX``-ready note projects, and optional PDF-build workflows.

While the numerical core of `Maranatha.jl` computes convergence data and
continuum fits, `Maranatha.Documentation.Reporter` supports the final presentation and archival stage
of the workflow.

---

## Example workflow

The example below demonstrates a minimal end-to-end workflow
using a configuration file and a simple integrand definition.

### Step 1 :: define an integrand

First define a small integrand in a Julia source file

Example integrand file (`sample_1d.jl`):

```julia
integrand(x) = sin(x)
```

---

### Step 2 :: prepare configuration file

Next prepare a configuration file describing the integration
domain, sampling sequence, quadrature rule, and output options.

The domain may be specified using scalar endpoints (for ``[a,b]^n``)
or axis-wise endpoints for rectangular regions.

Example configuration file (`sample_1d.toml`):

```toml
[integrand]
file = "sample_1d.jl"
name = "integrand"

[domain]
a = 0.0
b = 3.14159265358979323846264338327950588
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
real_type = "Double64"

[output]
name_prefix = "sample_1d"
save_path = "jld2"
write_summary = true
save_file = true
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

Once the dataset has been generated, the continuum limit ``h \to 0`` can be estimated by performing a least-``\chi^2`` fit.

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

### Execution backend selection

Execution backend is determined as follows:

- **CUDA backend** is used when `use_cuda = true`.
- **Threaded subgrid backend** is used when
  `use_cuda = false` and multiple Julia threads are available.
- Otherwise, serial execution is used.

The number of CPU threads is controlled by the environment variable
`JULIA_NUM_THREADS` set before starting Julia.

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