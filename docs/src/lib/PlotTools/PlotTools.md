# Maranatha.PlotTools

`Maranatha.PlotTools` contains the visualization layer of `Maranatha.jl`.

In a typical workflow, plotting appears after dataset generation and, when
relevant, after least-$\chi^2$ fitting:

1. build a convergence dataset with [`Maranatha.Runner.run_Maranatha`](@ref)
2. fit the extrapolation model with [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
3. visualize the result or inspect the raw datapoints with [`Maranatha.PlotTools`](@ref)

The module also provides a pedagogical 1D rule-coverage plot that is useful for
intuition-building and debugging, rather than for extrapolation itself.

---

## Overview

The plotting layer currently contains four main roles:

| Function | Responsibility |
|:--|:--|
| [`Maranatha.PlotTools.set_pyplot_latex_style`](@ref) | Configure global [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) / [`matplotlib`](https://matplotlib.org/stable/) styling for publication-like figures |
| [`Maranatha.PlotTools._smart_text_placement!`](@ref) | Place annotation boxes while heuristically avoiding plotted objects |
| [`Maranatha.PlotTools.plot_convergence_result`](@ref) | Visualize fitted extrapolation results and relative convergence error |
| [`Maranatha.PlotTools.plot_datapoints_result`](@ref) | Inspect raw datapoints before fitting |
| [`Maranatha.PlotTools.plot_quadrature_coverage_1d`](@ref) | Show how a selected $1$-dimensional quadrature rule samples or reconstructs the integrand |

---

## Internal Helpers

### [`Maranatha.PlotTools.set_pyplot_latex_style`](@ref)

This helper configures global [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) / [`matplotlib`](https://matplotlib.org/stable/) rendering parameters so that
plots are visually consistent across the package.

It enables [$\LaTeX$](https://www.latex-project.org/)-style text rendering and adjusts items such as:

- font sizes
- line widths
- marker sizes
- figure-scale-dependent layout settings

Because it modifies global plotting state, it is typically called internally by
the higher-level plotting routines before generating a figure.

### [`Maranatha.PlotTools._smart_text_placement!`](@ref)

This is an internal plotting helper used to place an annotation box inside an
axis while reducing visual collisions with plotted content.

The function evaluates a list of candidate anchor positions in axis-fraction
coordinates and scores each candidate after rendering, using display-space
geometry rather than data-space geometry. This makes the placement depend on the
actual rendered bounding box of the text and the actual screen-space position of
curves, markers, and error bars.

The scoring heuristic considers three main kinds of plotted objects:

1. discrete data points
2. polyline curves
3. vertical error bars

Candidates are penalized when the text box:

- extends outside the axis region
- overlaps a point
- intersects an error bar
- intersects a curve segment
- passes too close to plotted content

If several candidates are similarly good, the heuristic applies only a weak
preference for top or bottom placements over exact middle placements. The chosen
text is then drawn using a semi-transparent rounded white box for readability.

This helper is used by:

- [`Maranatha.PlotTools.plot_convergence_result`](@ref)
- [`Maranatha.PlotTools.plot_datapoints_result`](@ref)
- [`Maranatha.PlotTools.plot_quadrature_coverage_1d`](@ref)

---

## Public API

### [`Maranatha.PlotTools.plot_convergence_result`](@ref)

This routine visualizes a completed convergence fit. It is the natural plotting
companion to [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

It accepts raw convergence data together with a previously computed fit result
and reconstructs the fitted model directly from the stored fit object. No new
fit is performed during plotting.

#### What it plots

Two figures are generated:

1. a main convergence plot of $I(h)$ versus $h^p$
2. a log-log plot of the relative extrapolation error

The exponent $p$ is taken from the first non-constant stored fit power, i.e.
from `fit_result.powers[2]`. This means the horizontal axis is expressed in the
leading fitted convergence coordinate rather than in raw $h$.

#### Model reconstruction

The plotting stage uses the stored fit basis and covariance directly from
`fit_result`:

- `fit_result.params`
- `fit_result.cov`
- `fit_result.estimate`
- `fit_result.error_estimate`
- `fit_result.powers`

The reconstructed model is

```math
I(h) = \bm{\lambda}^{\mathsf{T}} \varphi(h),
\qquad
\varphi_\texttt{1}(h)=1,
\qquad
\varphi_\texttt{i}(h)=h^{\texttt{powers[i]}} \quad (\texttt{i} \ge 2).
```

This guarantees that the plotted curve is consistent with the actual fit basis,
including any forward-shifted exponent choice used during fitting.

#### Main convergence plot

The first figure contains:

- the reconstructed fit curve
- a propagated $1 \, \sigma$ fit band
- measured datapoints with pointwise error bars
- the extrapolated point at $h^p = 0$

The fit band is propagated from the parameter covariance:

```math
\sigma_{\mathrm{fit}}(h)^2 = \varphi(h)^{\mathsf{T}} V \varphi(h),
```

where $V = \texttt{fit\_result.cov}$.

The annotation box, containing $I_0$ and $\chi^2/\mathrm{d.o.f.}$, is placed by
[`Maranatha.PlotTools._smart_text_placement!`](@ref) so that it avoids the curve, markers, and error bars
when possible.

#### Relative-error plot

The second figure shows

```math
r(h) = \frac{|I(h)-I_0|}{|I_0|},
\qquad x = h^p,
```

on log-log axes.

The fitted relative-error curve is reconstructed as

```math
r_{\mathrm{fit}}(h) = \frac{|I_{\mathrm{fit}}(h)-I_0|}{|I_0|}.
```

For the measured datapoints, the routine propagates the error bars using the
pointwise uncertainty in $I(h)$ together with the uncertainty in $I_0$. The fit
band is propagated analogously from the covariance-propagated uncertainty of the
reconstructed curve.

A dashed slope-$1$ reference line is also drawn in the $h^p$ coordinate. In that
coordinate, a slope-1 line corresponds to the expected leading-order behavior
$r \propto h^p$.

#### Saved files

When `save_file = true`, the routine writes

```julia
result_$(name)_$(rule)_$(boundary)_extrap.pdf
result_$(name)_$(rule)_$(boundary)_reldiff.pdf
```

under `figs_dir`.

If `pdfcrop` is available, the saved PDFs are cropped automatically.

#### Convenience wrapper

A second method accepts the `result` object returned by [`Maranatha.Runner.run_Maranatha`](@ref) and
extracts:

- `result.a`
- `result.b`
- `result.h`
- `result.avg`
- `result.err`

before forwarding them to the primary plotting method.

This wrapper is useful when the user wants to plot directly from a stored or
freshly generated Maranatha result object without manual unpacking.

### [`Maranatha.PlotTools.plot_datapoints_result`](@ref)

This routine is a pre-fit diagnostic plotter.

Unlike [`Maranatha.PlotTools.plot_convergence_result`](@ref), it does **not** use a fitted model. It simply
plots the sampled datapoints and their error bars so the user can inspect the
raw convergence pattern before deciding how to fit it.

#### Purpose

This is useful for checking whether the convergence data:

- aligns roughly linearly in a chosen transformed coordinate
- shows oscillatory behavior between resolutions
- contains suspicious outliers
- looks suitable for a particular fit-power choice

#### Horizontal coordinate

The horizontal coordinate is chosen manually as

```math
x = h^p,
```

where `p = h_power` is supplied by the caller.

This makes it easy to inspect whether a candidate power of $h$ produces an
approximately straight trend before running a least-$\chi^2$ extrapolation.

#### Axis scaling

The routine supports independent axis scaling:

- `xscale = :linear` or `:log`
- `yscale = :linear` or `:log`

Datapoints incompatible with the chosen scaling are filtered out before
plotting.

#### Annotation

The figure includes a small annotation box containing the chosen axis scales and
the selected `h_power`. Its position is selected with
[`Maranatha.PlotTools._smart_text_placement!`](@ref).

#### Saved files

When `save_file = true`, the routine writes

```julia
result_$(name)_$(rule)_$(boundary)_datapoints_hpow_$(h_power)_$(xscale)_$(yscale).pdf
```

under `figs_dir`, with optional `pdfcrop` post-processing if available.

#### Convenience wrapper

As with [`Maranatha.PlotTools.plot_convergence_result`](@ref), a wrapper method accepts the standard
Maranatha `result` object and forwards its stored fields to the primary method.

### [`Maranatha.PlotTools.plot_quadrature_coverage_1d`](@ref)

This routine is a pedagogical and debugging-oriented visualizer for $1$-dimensional
quadrature rules.

It is fundamentally different from the fit-oriented plotters above. Instead of
showing convergence across resolutions, it shows how a **single selected 1D
rule** samples or reconstructs the integrand on $[a,b]$.

#### What it always draws

The routine always draws:

1. a dense reference curve of the true integrand $f(x)$
2. the quadrature nodes and weights returned by the 1D dispatcher
3. a quadrature-sum annotation

The quadrature-sum label is automatically positioned by
[`Maranatha.PlotTools._smart_text_placement!`](@ref).

#### B-spline rules

For B-spline rules, the routine reconstructs the effective spline curve
implicitly used by the backend:

- sample $f(x)$ at the quadrature node set
- solve the corresponding spline reconstruction problem
- evaluate the spline piecewise over knot spans
- draw and fill the reconstructed spline per span

This makes the plot reflect the effective curve that the B-spline quadrature
backend is actually integrating, rather than only the raw sample values.

#### Non-B-spline rules

For Newton-Cotes and Gauss-family rules, the routine switches to a schematic
mass-bar interpretation.

Each quadrature contribution $w_i \, f(x_i)$ is shown as a rectangular block whose:

- width is $w_i$
- height is $f(x_i)$

so that the signed area equals the signed quadrature contribution.

The bars are drawn sequentially from $a$. Their horizontal positions are
therefore pedagogical rather than literal node positions. This is intentional:
the purpose is to visualize signed contribution structure clearly with minimal
assumptions.

If a weight is negative, the drawn interval is flipped so that the displayed bar
still runs left-to-right while preserving the correct signed area.

#### Rule-family detection

Rule-family dispatch is determined internally using:

- [`Maranatha.Quadrature.NewtonCotes._is_newton_cotes_rule`](@ref)
- [`Maranatha.Quadrature.Gauss._is_gauss_rule`](@ref)
- [`Maranatha.Quadrature.BSpline._is_bspline_rule`](@ref)

The `(rule, boundary)` pair is forwarded unchanged to the quadrature backend.

#### Saved file

When `save_file = true`, the routine writes

```julia
pedagogical_1D_$(name)_$(rule)_$(boundary)_N$(N).pdf
```

under `figs_dir`, again using `pdfcrop` when available.

#### [`TOML`](https://toml.io/en/) wrapper

A second method accepts a Maranatha [`TOML`](https://toml.io/en/) file.

That wrapper:

1. parses the [`TOML`](https://toml.io/en/) configuration
2. validates it
3. loads the integrand from the referenced Julia file
4. dispatches to the primary plotting method

If `N` is provided, only that subdivision count is plotted. If `N === nothing`,
the wrapper plots every `N` listed in `cfg.nsamples`.

Because the integrand is loaded dynamically, the wrapper uses
`Base.invokelatest` when forwarding to the primary plotting routine.

---

## API reference

```@autodocs
Modules = [
    Maranatha.PlotTools,
]
Private = true
```