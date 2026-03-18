# Maranatha.Runner

[`Maranatha.Runner`](@ref) provides the dataset-construction layer of the package.
Its job is not to perform the final ``h \to 0`` fit, but to build the raw
multi-resolution convergence data that the fitter and plotter consume later.

---

## Responsibility in the full workflow

| Stage | Responsibility |
|:------|:---------------|
| Runner | builds raw convergence datasets |
| Quadrature | supplies tensor-product quadrature in arbitrary dimensions |
| Error estimator | supplies derivative-based or refinement-based error-scale estimators |
| Fitter | performs ``h \to 0`` extrapolation |
| Plotter | visualizes convergence |
| Reporter | generates structured summaries and archival report outputs |

In other words, the runner is the front door of a typical computational
workflow in `Maranatha.jl`.
You define an integrand and a sequence of resolutions, and the runner returns a
uniform dataset containing step sizes, quadrature estimates, and modeled error
scales.

---

## What belongs in the API docstring vs. this page

The docstring for [`run_Maranatha`](@ref) should focus on API-level material:

- function signature,
- arguments and keyword arguments,
- return structure,
- saving behavior,
- short usage examples.

This page is a better place for broader narrative material such as:

- the role of the runner inside the package architecture,
- the standard end-to-end workflow,
- [`TOML`](https://toml.io/en/)-oriented usage patterns,
- pedagogical explanations and longer examples.

That keeps the in-source docstring readable in the `REPL` while still allowing the
manual page to provide fuller context.

---

## Standard workflow

A typical workflow is:

1. call [`run_Maranatha`](@ref) to generate a raw convergence dataset,
2. pass the returned result to
   [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref),
3. inspect or visualize the fit using
   [`Maranatha.Documentation.PlotTools.plot_convergence_result`](@ref),
4. optionally generate structured reports using
   [`Maranatha.Documentation.Reporter.write_convergence_internal_note`](@ref).

The runner itself does not fit the continuum-limit model.
It only prepares the data needed for later stages of analysis,
visualization, and reporting.

At the start of each run, the runner clears internal error-estimation caches
to avoid cross-run contamination when multiple datasets are generated in a
single Julia session.

Depending on the selected configuration, the error model may be computed
either through a refinement-based estimator or through a derivative-based
estimator.

When a derivative-based estimator is used, the implementation may evaluate
derivatives either directly or through jet-based reuse, depending on the
current backend configuration.

---

## Direct-call usage

```julia
using Maranatha

f(x) = sin(x)

run_result = run_Maranatha(
    f,
    0.0,
    pi;
    dim = 1,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :refinement,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_error_jet = false,
)
```

In current versions, the error-estimation backend may internally use either
a refinement-based path or a derivative-based path.

When a derivative-based estimator is selected, it may further use a jet-based
derivative path for improved performance. These are implementation details and
do not change the external API.

### Execution backend selection

`run_Maranatha` automatically selects the quadrature execution backend:

- **CUDA backend** is used when `use_cuda = true`.
- **Threaded subgrid backend** is used when
  `use_cuda = false` and Julia is started with multiple threads.
- Otherwise, serial execution is used.

The number of CPU threads is controlled by the environment variable
`JULIA_NUM_THREADS`, which must be set before starting Julia.

---

## [`TOML`](https://toml.io/en/)-based usage

The runner can also be used through a configuration file.
A small example is shown below.

### Example integrand file

```julia
integrand(x) = sin(x)
```

### Example [`TOML`](https://toml.io/en/) file

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

You can then run:

```julia
using Maranatha

run_result = run_Maranatha("./sample_1d.toml")
```

The execution backend is determined by runtime settings.
CUDA execution requires explicit enabling in the API, while CPU
parallelism depends on the number of Julia threads.

---

## What the runner returns

The returned object is a `NamedTuple` designed to be passed downstream.
The most commonly used fields are:

- `result.a`, `result.b`
- `result.h`
- `result.avg`
- `result.err`
- `result.rule`, `result.boundary`
- `result.fit_terms`, `result.ff_shift`

These fields are usually enough to continue directly into the fitting stage.

---

## Further examples

For detailed documentation, tutorials, and complete workflow examples,
please refer to the project documentation site or the example Jupyter notebooks
located in the `ipynb/` directory of this repository.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Runner,
]
Private = true
```