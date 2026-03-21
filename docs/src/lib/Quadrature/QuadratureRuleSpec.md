# Maranatha.Quadrature.QuadratureRuleSpec

`Maranatha.Quadrature.QuadratureRuleSpec` centralizes validation and normalization of
quadrature-rule specifications.

This module exists because modern `Maranatha.jl` workflows allow `rule` to be
supplied either as:

- a single scalar rule symbol shared by all axes, or
- a tuple / vector of per-axis rule symbols.

Keeping that logic in one place avoids repeated ad hoc validation across the
quadrature, error-estimation, runner, and reporting layers.

---

## Overview

The helpers in this module answer three main questions:

1. which quadrature family does a scalar rule belong to,
2. what scalar rule is active on a given axis,
3. whether a full scalar-or-axis-wise rule specification is valid for a given
   problem dimension.

They also provide a same-family check used by the refinement-based
error-estimation branch.

---

## Supported families

The current rule families recognized by `QuadratureRuleSpec` are:

- Newton-Cotes: `:newton_p*`
- Gauss-family: `:gauss_p*`
- B-spline: `:bspline_interp_p*`, `:bspline_smooth_p*`

Family-specific parsing and numerical construction remain in the dedicated
backend modules. `QuadratureRuleSpec` only manages the specification layer.

---

## Key helpers

| Helper | Responsibility |
|:--|:--|
| [`Maranatha.Quadrature.QuadratureRuleSpec._rule_family`](@ref) | classify one resolved scalar rule symbol by family |
| [`Maranatha.Quadrature.QuadratureRuleSpec._rule_at`](@ref) | resolve the scalar rule used on one axis |
| [`Maranatha.Quadrature.QuadratureRuleSpec._validate_rule_spec`](@ref) | validate a scalar-or-axis-wise rule specification |
| [`Maranatha.Quadrature.QuadratureRuleSpec._has_axiswise_rule_spec`](@ref) | detect whether a specification is genuinely axis-wise |
| [`Maranatha.Quadrature.QuadratureRuleSpec._common_rule_family`](@ref) | verify that all axes belong to one rule family |
| [`Maranatha.Quadrature.QuadratureRuleSpec._rule_spec_string`](@ref) | build a compact human/file-friendly string representation |

---

## Why this module matters

Without `QuadratureRuleSpec`, every caller that accepts `rule` would need to repeat:

- scalar-versus-axis-wise checks,
- dimension checks,
- family checks,
- formatting logic.

That would be easy to get wrong, especially now that:

- quadrature supports axis-wise `rule`,
- derivative-based error estimation supports axis-wise `rule`,
- refinement supports axis-wise `rule` only under the same-family restriction,
- filenames and report tables must reflect either scalar or per-axis rules.

---

## Relationship to `QuadratureBoundarySpec`

`QuadratureRuleSpec` is the rule-side counterpart of
[`Maranatha.Utils.QuadratureBoundarySpec`](@ref).

Together they provide the normalized specification layer that higher-level
modules use before constructing nodes, residual models, fits, or reports.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Quadrature.QuadratureRuleSpec,
]
Private = true
```
