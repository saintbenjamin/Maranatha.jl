# Maranatha.Utils.Reporter

`Maranatha.Utils.Reporter` provides high-level reporting utilities for
generating structured summaries of convergence studies performed with
`Maranatha.jl`.

Unlike low-level logging or plotting tools, this module focuses on
**publication-quality output artifacts**, including:

- formatted tables of convergence data,
- [``\LaTeX``](https://www.latex-project.org/)-ready summaries,
- complete internal-note projects,
- automated PDF generation workflows.

These tools are designed to make numerical experiments reproducible,
shareable, and suitable for research documentation.

---

## Overview

A typical `Maranatha.jl` workflow produces raw convergence data and, in many
cases, a fitted continuum extrapolation. The `Maranatha.Utils.Reporter` module
transforms either stage into human-readable reports.

```
Runner → LeastChiSquareFit → PlotTools → Reporter
```

For fit-based reporting, `Maranatha.Utils.Reporter` consumes both the quadrature result and the
fit result. For pre-fit reporting, it can also generate summaries and internal
notes directly from raw datapoints and datapoints-only plots.

The `Maranatha.Utils.Reporter` module is therefore the final stage of many pipelines.

---

## Core capabilities

### 1. Convergence summary tables

Reporter can generate both:

- **fit-aware convergence summaries**, including
  - step sizes,
  - estimates,
  - uncertainties,
  - fitted parameters,
  - extrapolated results,

and

- **datapoints-only summaries**, including
  - step sizes,
  - transformed horizontal coordinates such as ``h^p``,
  - raw quadrature estimates,
  - pointwise uncertainties,
  - plotting metadata such as axis-scale conventions.

Outputs can be produced in multiple formats, including [$\LaTeX$](https://www.latex-project.org/)-ready
snippets for direct inclusion in documents.

This allows Reporter to support both:
- post-fit reporting workflows, and
- pre-fit inspection / archival workflows based only on raw datapoints.

---

### 2. Internal note generation

The flagship feature is the ability to construct a **self-contained
[$\LaTeX$](https://www.latex-project.org/) project** representing a numerical experiment.

Reporter supports two related internal-note workflows:

- **fit-based internal notes**, which include fitted extrapolation summaries and
  convergence-fit figures,
- **datapoints-only internal notes**, which summarize raw quadrature datapoints
  and their plotting configuration without requiring a fit result.

Depending on the workflow, the generated note can include:

- a REVTeX-based master document
- formatted summary tables
- figure inclusion files
- a reproducible Makefile
- organized plot assets
- optional automatic PDF compilation

The resulting directory can be archived or shared as a complete research
artifact.

---

### 3. Automatic build workflow

Reporter can optionally compile the generated note into a PDF.

The build system:

- checks for required [$\LaTeX$](https://www.latex-project.org/) executables,
- verifies availability of required class and package files,
- uses `make` if present,
- falls back to direct `pdflatex` builds,
- handles bibliography processing when needed,
- reports missing dependencies with actionable diagnostics.

---

## Typical usage

A common usage pattern is:

```julia
using Maranatha

run_result = run_Maranatha("../samples/sample_1d.toml")

fit_result = least_chi_square_fit(
    run_result; 
    nterms=4, 
    ff_shift=0, 
    nerr_terms=3
)

plot_convergence_result(
    run_result, 
    fit_result;
    name="sample_1d",
    figs_dir=".",
    save_file=true
)

note_info = write_convergence_internal_note(
    run_result,
    fit_result;
    name = "sample_1d",
    rule = run_result.rule,
    boundary = run_result.boundary,
    out_dir = ".",
    save_file = true,
    try_build_pdf = true,
    move_existing_plots = true,
)
```

This produces a directory such as:

```
inote_summary_sample_1d_gauss_p4_LU_EXEX_ff_4_er_3/
    inote_summary_sample_1d_gauss_p4_LU_EXEX_ff_4_er_3.tex
    Makefile
    figs/
        result_sample_1d_gauss_p4_LU_EXEX_extrap.pdf
        result_sample_1d_gauss_p4_LU_EXEX_reldiff.pdf
```

A datapoints-only workflow is also supported when you want to document raw
quadrature behavior before fitting:

```julia
using Maranatha

run_result = run_Maranatha("../samples/sample_1d.toml")

plot_datapoints_result(
    run_result;
    name = "sample_1d",
    h_power = 2,
    xscale = :log,
    yscale = :linear,
    figs_dir = ".",
    save_file = true,
)

write_convergence_summary_datapoints(
    run_result;
    name = "sample_1d",
    h_power = 2,
    xscale = :log,
    yscale = :linear,
    format = :md,
    out_dir = ".",
    save_file = true,
)

note_info = write_convergence_internal_note_datapoints(
    run_result;
    name = "sample_1d",
    h_power = 2,
    xscale = :log,
    yscale = :linear,
    out_dir = ".",
    save_file = true,
    try_build_pdf = true,
    move_existing_plots = true,
)
```

This produces a datapoints-only note directory containing a raw-datapoint
summary and the corresponding saved datapoints figure.

---

## When to use `Maranatha.Utils.Reporter`

Use `Maranatha.Utils.Reporter` when you need:

- research-grade documentation of numerical results,
- shareable experiment summaries,
- reproducible [$\LaTeX$](https://www.latex-project.org/) output,
- archival records of convergence studies,
- automated generation of internal reports,
- pre-fit summaries of raw quadrature datapoints,
- datapoints-only internal notes for inspection and debugging.

It is especially useful for:

- papers and technical notes,
- collaboration artifacts,
- verification of numerical pipelines,
- debugging convergence behavior,
- reproducibility workflows.

---

## Design philosophy

Reporter emphasizes:

### Reproducibility

Every generated report contains enough information to rebuild the
document independently.

---

### Publication-quality output

Formatting is designed to integrate seamlessly into professional
[$\LaTeX$](https://www.latex-project.org/) documents, particularly REVTeX-based manuscripts common in
physics and computational science.

---

### Robustness

Build steps are defensive:

- missing files are detected early,
- dependency checks prevent opaque failures,
- fallbacks ensure maximum portability.

---

### Separation of concerns

Reporter does not perform numerical computations itself.
It operates purely on results produced by other modules.

---

## Output directory conventions

Generated note directories follow the pattern:

```
inote_$(summary_basename)/
```

Depending on whether the workflow is fit-based or datapoints-only, the directory
contains the corresponding summary and figure assets.

Typical contents include:

| Component | Purpose |
|----------|----------|
Master `.tex` | Stand-alone document |
Summary table | Numerical results or datapoints-only summary |
Figure include file | Plot layout |
`figs/` directory | Plot assets |
Makefile | Reproducible build |

---

## [$\LaTeX$](https://www.latex-project.org/) requirements

The default document template uses REVTeX (`revtex4-2`) and a set of
commonly available packages.

Reporter attempts to verify their presence before building. If tools
such as `kpsewhich` are unavailable, validation is deferred to the
[$\LaTeX$](https://www.latex-project.org/) engine.

---

## Relationship to other modules

Reporter complements:

- [`Maranatha.Runner`](@ref) — generates convergence datasets
- [`Maranatha.LeastChiSquareFit`](@ref) — computes extrapolations
- [`Maranatha.PlotTools`](@ref) — produces visualization artifacts

Reporter consumes the outputs of these modules to produce final reports.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.Reporter,
]
Private = true
```