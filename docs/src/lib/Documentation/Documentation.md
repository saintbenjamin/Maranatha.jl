# Maranatha.Documentation

`Maranatha.Documentation` provides the output layer of `Maranatha.jl`,
combining visualization tools and report-generation utilities for
convergence studies.

In a typical workflow, numerical results are produced by computational
modules and then transformed into human-readable artifacts by this layer.

```
Computation Layer
    ├─ Quadrature
    ├─ ErrorEstimate
    ├─ Runner
    └─ LeastChiSquareFit

Documentation Layer (post-processing)
    └─ Documentation
        ├─ PlotTools
        └─ Reporter
```

The module does not perform numerical integration or fitting itself.
Instead, it consumes the outputs of other components and produces
publication-ready figures, summaries, and reproducible report projects.

---

## Overview

`Maranatha.Documentation` currently consists of two complementary submodules:

| Submodule | Responsibility |
|:--|:--|
| [`Maranatha.Documentation.PlotTools`](@ref) | Visualization of convergence results, raw datapoints, and rule behavior |
| [`Maranatha.Documentation.Reporter`](@ref) | Structured summaries and self-contained report generation |

Together, these tools form the final stage of most `Maranatha.jl` pipelines.

---

## Plotting layer — `PlotTools`

[`Maranatha.Documentation.PlotTools`](@ref) provides visualization routines for
both fitted and pre-fit workflows.

Typical uses include:

- plotting extrapolation curves and uncertainty bands,
- inspecting raw convergence datapoints before fitting,
- generating figures suitable for publications,
- visualizing how quadrature rules sample an integrand.

Plots are produced using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) /
[`matplotlib`](https://matplotlib.org/stable/) with styling tuned for
LaTeX-compatible output.

The plotting layer is especially useful for exploratory analysis,
diagnostics, and figure generation for papers or technical reports.

---

## Reporting layer — `Reporter`

[`Maranatha.Documentation.Reporter`](@ref) provides high-level tools for
generating structured documentation of numerical experiments.

Capabilities include:

- formatted convergence summaries,
- LaTeX- and Markdown-ready tables,
- pre-fit datapoint reports,
- complete internal-note projects,
- automated PDF build workflows.

Unlike plotting utilities, Reporter focuses on producing **shareable
research artifacts** rather than visual diagnostics alone.

The generated outputs are designed to be reproducible and suitable for
archival or collaboration.

---

## Typical workflow

A common sequence is:

1. Generate a convergence dataset:

```julia
run_result = run_Maranatha("config.toml")
```

2. (Optional) Fit an extrapolation model:

```julia
fit_result = least_chi_square_fit(run_result)
```

3. Produce figures:

```julia
plot_convergence_result(run_result, fit_result; name="experiment")
```

4. Generate a report or internal note:

```julia
write_convergence_internal_note(
    run_result,
    fit_result;
    name="experiment",
    save_file=true,
    try_build_pdf=true,
)
```

For pre-fit analysis, plotting and reporting can also operate directly on
raw datapoints.

---

## Design principles

### Separation of concerns

`Maranatha.Documentation` performs no numerical computation.
It operates purely on results produced by other modules.

---

### Reproducibility

Outputs contain sufficient information to reconstruct figures or documents
independently of the original computation.

---

### Publication-quality artifacts

Figures and reports are formatted for seamless inclusion in professional
LaTeX documents and technical manuscripts.

---

### Complementary tooling

Plotting and reporting tools address different needs:

* **PlotTools** — visual diagnostics and figure generation
* **Reporter** — structured summaries and complete report projects

Together, they provide a comprehensive documentation layer for numerical
studies.

---

## Relationship to other modules

`Maranatha.Documentation` consumes outputs from:

* [`Maranatha.Runner`](@ref) — convergence dataset construction
* [`Maranatha.LeastChiSquareFit`](@ref) — extrapolation fitting
* [`Maranatha.Quadrature`](@ref) — quadrature rule definitions

It represents the final stage in many computational workflows.

---

## When to use `Maranatha.Documentation`

Use this module when you need:

* publication-ready figures,
* diagnostic plots of convergence behavior,
* structured experiment summaries,
* reproducible LaTeX output,
* shareable internal reports,
* archival records of numerical studies.

---

## API reference

```@autodocs
Modules = [
    Maranatha.Documentation,
]
Private = true
```