# Maranatha.LeastChiSquareFit

`Maranatha.LeastChiSquareFit` performs weighted ``h \to 0`` extrapolation from raw
convergence datasets produced by [`Maranatha.Runner.run_Maranatha`](@ref).

In a standard workflow:

1. build a dataset with [`Maranatha.Runner.run_Maranatha`](@ref)
2. fit the convergence model with [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
3. inspect the fitted parameters with [`Maranatha.LeastChiSquareFit.print_fit_result`](@ref)
4. optionally visualize the result with
   [`Maranatha.PlotTools.plot_convergence_result`](@ref)

---

## Overview

The fitter assumes a convergence model that is linear in its coefficients:

```math
I(h) = I_0 + C_1 h^{p_1} + C_2 h^{p_2} + \cdots
```

Here, `I_0` is the extrapolated continuum-limit estimate, and the exponents
`p_1, p_2, ...` are inferred automatically from the midpoint-residual structure
associated with the chosen quadrature rule and boundary pattern.

The module provides:

- a main fitting entry point that accepts raw arrays
- a convenience overload that accepts a Maranatha runner result object
- a compact printer for fit summaries

---

## Main fitting routine

The primary method is:

```julia
least_chi_square_fit(a, b, hs, estimates, error_infos, rule, boundary; ...)
```

It performs the following steps.

### 1. Infer candidate residual powers

A representative subdivision count is inferred from the smallest supplied step size:

```math
N_{\mathrm{ref}} = \mathrm{round}\!\left(\frac{b-a}{\min(h)}\right).
```

The midpoint-residual backend is then queried through
[`Maranatha.ErrorEstimate.ErrorDispatch._leading_residual_ks_with_center_any`](@ref), producing a list of
candidate residual indices `ks`.

### 2. Map residual indices to fit powers

The raw residual indices are converted into fit exponents depending on the rule family.

- **Newton-Cotes rules**: `powers_all = ks`
- **Gauss-family rules**: `powers_all = ks`
- **B-spline rules**: `powers_all = ks .+ 1`

After that, the implementation normalizes the list defensively:

- remove duplicates,
- sort the powers,
- discard nonpositive entries so that the constant intercept column is not duplicated.

### 3. Apply forward shift

If `nterms` includes the constant term, then the fitter needs exactly
`need = nterms - 1` nonconstant powers.

With forward shift `ff_shift`, the selected powers are:

```math
\texttt{powers} = \texttt{powers\_all[(1+ff\_shift):(1+ff\_shift+need-1)]}.
```

This is useful when the nominal leading-order coefficient vanishes for the chosen
integrand, so a more stable fit may be obtained by skipping the first candidate power.

### 4. Build the design matrix

The fitted model is linear in the parameters:

```math
I(h) = I_0 + C_1 h^{p_1} + C_2 h^{p_2} + \cdots + C_{m} h^{p_m}.
```

The design matrix therefore has:

- column `1`: the constant term `1`
- column `t+1`: `h^(powers[t])`

### 5. Construct the uncertainty vector

The fitter does not use the error-info objects directly as a scalar uncertainty.
Instead, it reconstructs an effective uncertainty vector `σ` from the stored residual terms.

For each entry in `error_infos`, it takes the first `nerr_terms` residual contributions and uses

```math
\sigma_i = \left| \sum_{m=1}^{n_{\mathrm{err}}} \texttt{terms}_m \right|.
```

This matches the runner-side interpretation of the stored residual model.

### 6. Solve the weighted least-squares problem

Let `X` be the design matrix, `y` the estimate vector, and

```math
W = \mathrm{diag}(1/\sigma).
```

The implementation solves the weighted least-squares system

```math
(WX)\,\lambda = Wy,
```

using Julia's backslash solver.

### 7. Estimate the covariance

The implementation builds

```math
A = X^{\mathsf T} W^2 X,
\qquad
H = 2A,
```

then computes a Cholesky factorization of the Hessian:

```math
H = LL^{\mathsf T}.
```

The covariance is then assembled through linear solves rather than explicit matrix inversion.
The returned parameter uncertainties are

```math
\sqrt{\operatorname{diag}(\mathrm{Cov})}.
```

### 8. Return fit diagnostics

The returned `NamedTuple` stores:

- `estimate`
- `error_estimate`
- `params`
- `param_errors`
- `cov`
- `powers`
- `chisq`
- `redchisq`
- `dof`

The stored `powers` field includes the constant term as `vcat(0, powers)` so that
plotting and reporting utilities can reconstruct the fitted model without repeating
power inference.

---

## Convenience overload

The second method accepts a Maranatha result object directly:

```julia
least_chi_square_fit(result; nterms=nothing, ff_shift=nothing, nerr_terms=nothing)
```

If a keyword is omitted, the corresponding value already stored in `result` is reused.
This makes it convenient to write:

```julia
fit_result = least_chi_square_fit(run_result)
```

or to override only one downstream fit setting:

```julia
fit_result = least_chi_square_fit(run_result; ff_shift=1)
```

---

## Printed summary

[`print_fit_result`](@ref) prints:

- each fitted parameter `λ_k` with its uncertainty,
- the total ``\chi^2`` and reduced ``\chi^2``,
- the extrapolated continuum value `I(h \to 0)`.

The output is intentionally compact and log-friendly.

## Errors and edge cases

The main fitting routine can throw if:

- `nterms < 2`
- `ff_shift < 0`
- not enough candidate powers remain after applying `ff_shift`
- the effective uncertainty vector contains nonpositive entries
- the Hessian is not positive definite, so the Cholesky factorization fails

Also note that if `dof == 0`, then `redchisq = chisq / dof` follows IEEE rules and may
become `Inf` or `NaN`. In that case the fit may still produce parameters, but reduced
``\chi^2`` is no longer a meaningful diagnostic.

---

## Typical workflow example

```julia
using Maranatha

run_result = run_Maranatha("./sample_1d.toml")

fit_result = least_chi_square_fit(
    run_result;
    nterms = 3,
    ff_shift = 0,
    nerr_terms = 2,
)

print_fit_result(fit_result)

plot_convergence_result(
    run_result,
    fit_result;
    name = "Maranatha_test1",
    figs_dir = ".",
    save_file = true,
)
```

For fuller tutorials and interactive demonstrations, see the project documentation
site and the example Jupyter notebooks in the `ipynb/` directory of this repository.

---

## API reference

```@autodocs
Modules = [
    Maranatha.LeastChiSquareFit,
]
Private = true
```