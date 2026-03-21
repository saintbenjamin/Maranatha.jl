# Maranatha.LeastChiSquareFit

`Maranatha.LeastChiSquareFit` performs weighted `h \to 0` extrapolation from
the convergence datasets produced by [`Maranatha.Runner.run_Maranatha`](@ref).

In a standard workflow:

1. build a dataset with [`Maranatha.Runner.run_Maranatha`](@ref),
2. fit the extrapolation model with
   [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref),
3. inspect the fitted parameters with
   [`Maranatha.LeastChiSquareFit.print_fit_result`](@ref),
4. optionally visualize or report the result.

---

## Overview

The fitter assumes a linear-in-parameters convergence model

```math
I(h) = I_0 + C_1 h^{p_1} + C_2 h^{p_2} + \cdots ,
```

where:

- `I_0` is the extrapolated continuum-limit estimate,
- the nonconstant powers are inferred automatically from the rule-dependent
  residual structure,
- the fit is performed on the scalar step proxy stored in `result.h`.

The module provides:

- one main fitting entry point operating on a runner result object,
- a compact printer for fit summaries and diagnostics.

---

## Main fitting routine

The primary public method is:

```julia
least_chi_square_fit(
    result;
    fit_func_terms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms,
)
```

It works directly on the `NamedTuple` returned by
[`Maranatha.Runner.run_Maranatha`](@ref).

---

## How power selection works

### 1. Choose a representative subdivision count

The fitter starts from

```julia
Nref = minimum(result.nsamples)
```

rather than inferring a subdivision count geometrically from `a`, `b`, and `h`.

If any active axis uses a Newton-Cotes rule, this `Nref` is increased to the
smallest subdivision count that is simultaneously admissible for all active
Newton-Cotes axes.

### 2. Resolve per-axis rule and boundary data

The fitter accepts both scalar and axis-wise `rule` / `boundary`
specifications.

Internally, it resolves the scalar rule and boundary used on each axis before
querying the residual-power model.

### 3. Collect residual powers axis by axis

For each active axis, the fitter queries
[`Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative._leading_residual_ks_with_center_any`](@ref)
to obtain the leading residual indices compatible with that axis-local rule and
boundary.

If the problem is axis-wise, the per-axis candidate lists are merged by taking
their sorted union.

### 4. Apply the forward shift

After duplicate removal and sorting, the fitter selects the nonconstant powers
by applying the user-controlled forward shift `ff_shift`.

This makes it possible to skip nominal leading powers when:

- the leading coefficient vanishes for the chosen integrand,
- pre-asymptotic contamination is strong,
- a cleaner fit is obtained by starting one or more powers later.

---

## Weighted least-squares system

Once the powers are chosen, the fitter constructs the design matrix

```math
X = [1,\ h^{p_1},\ h^{p_2},\ \ldots ] .
```

The response vector is taken from `result.avg`, and the uncertainty vector `σ`
is reconstructed from `result.err`.

### Uncertainty extraction

The helper
[`Maranatha.LeastChiSquareFit._extract_sigma_from_error_info`](@ref)
supports two stored error-info layouts:

- derivative-based residual objects:
  use the first `nerr_terms` entries of `e.terms`,
- refinement-based error objects:
  use `abs(e.estimate)` directly.

The weighted least-squares problem is then solved in the usual form

```math
(WX)\lambda = Wy,
\qquad
W = \operatorname{diag}(1/\sigma).
```

The module also estimates the parameter covariance and returns parameter
uncertainties together with `χ²` diagnostics.

---

## Result object

The returned `NamedTuple` contains:

- `estimate`
- `error_estimate`
- `params`
- `param_errors`
- `cov`
- `powers`
- `chisq`
- `redchisq`
- `dof`
- `fit_func_terms`
- `nerr_terms`

The stored `powers` field includes the constant term as the leading `0`, so
plotting and reporting layers can reconstruct the fitted model without
repeating exponent selection.

---

## Practical notes

- The fitter works on the scalarized step proxy `result.h`.
  Exact per-axis step objects remain available in `result.tuple_h`, but they
  are not fitted directly.
- Axis-wise `rule` / `boundary` specifications are supported naturally through
  the merged residual-power workflow.
- `ff_shift` is a practical tuning knob for cases where the nominal leading
  power is not the most stable basis choice on the sampled refinement ladder.

---

## Errors and edge cases

The main fitting routine can fail when:

- `fit_func_terms < 2`,
- `ff_shift < 0`,
- not enough candidate powers remain after applying `ff_shift`,
- the effective uncertainty vector contains invalid entries,
- the weighted Hessian is not positive definite.

Also note that when `dof == 0`, the reduced `χ²` follows IEEE division rules
and may become `Inf` or `NaN`.

---

## Printed summary

[`Maranatha.LeastChiSquareFit.print_fit_result`](@ref) prints a compact summary
of:

- the extrapolated continuum value,
- the fitted parameters and uncertainties,
- `χ²` and reduced `χ²`.

This is intended for logs, notebooks, and quick inspection rather than for
full report generation.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.LeastChiSquareFit,
]
Private = true
```
