# ============================================================================
# src/Documentation/Reporter/_build_convergence_summary_basename.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_convergence_summary_basename(
        name,
        rule,
        boundary,
        fit_terms,
        nerr_terms,
    ) -> String

Construct a standardized base filename for convergence-summary outputs.

# Function description

This helper builds a descriptive basename encoding key run parameters,
including the integrand name, quadrature rule, boundary condition,
and fitting configuration.

The result is typically used as the stem for output files such as:

- [``\\LaTeX``](https://www.latex-project.org/) summary tables
- Markdown reports
- auxiliary artifacts

Optional suffixes are appended only when the corresponding values are not
`nothing`.

# Arguments

- `name`: Identifier for the integrand or experiment.
- `rule`: Quadrature rule used.
- `boundary`: Boundary-handling scheme.
- `fit_terms`: Number of terms used in the extrapolation fit.
- `nerr_terms`: Number of error terms included in the model.

# Returns

- `String`: A filesystem-friendly basename.

# Notes

- The function performs no sanitization beyond string conversion.
- Callers are responsible for ensuring the result is valid as a filename.
"""
function _build_convergence_summary_basename(
    name::AbstractString,
    rule,
    boundary,
    fit_terms,
    nerr_terms,
)
    base = "summary_$(name)_$(String(rule))_$(String(boundary))"

    fit_terms_suffix = isnothing(fit_terms) ? "" : "_ff_$(fit_terms)"
    nerr_terms_suffix = isnothing(nerr_terms) ? "" : "_er_$(nerr_terms)"

    return base * fit_terms_suffix * nerr_terms_suffix
end