# ============================================================================
# src/Documentation/Reporter/_build_fit_model_tex.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_fit_model_tex(fit_powers) -> String

Construct a [``\\LaTeX``](https://www.latex-project.org/)-ready polynomial model expression for the convergence fit.

# Function description

This helper generates a symbolic model of the form
```math
f(h) = \\lambda_0 + \\lambda_1 h^{p_1} + \\lambda_2 h^{p_2} + \\ldots
```
using the sequence of powers supplied in `fit_powers`.

Each coefficient is labeled as ``\\lambda_i`` (starting from ``i = 0``), and the
corresponding power of ``h`` is taken from the input vector.

Special cases are handled for readability:

- power ``0``: constant term ``\\lambda_0``
- power ``1``: linear term ``\\lambda_1 h``
- power ``p > 1``: general term ``\\lambda_i h^{p}``

The result is returned as a plain [``\\LaTeX``](https://www.latex-project.org/)-compatible string without enclosing
math delimiters.

# Arguments

- `fit_powers`: Iterable collection of exponents used in the fit model.
  Typically obtained from a least-χ² extrapolation routine.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/)-formatted model expression suitable for captions,
  tables, or inline text.

# Errors

- No explicit validation is performed.
- Non-numeric entries in `fit_powers` will propagate to the output string.

# Notes

- The function assumes that the order of `fit_powers` matches the order of
  fitted parameters.
- No simplification or sorting is applied.
- The output string does not include `\$` math-mode delimiters, allowing the
  caller to choose inline or display math contexts.
- Intended for reporting and documentation rather than symbolic algebra.

# Examples

Typical output:

    f(h) = \\lambda_0 + \\lambda_1 h^2 + \\lambda_2 h^4

when `fit_powers = [0, 2, 4]`.
"""
function _build_fit_model_tex(fit_powers)
    terms = String[]

    for (i, p) in enumerate(fit_powers)
        λ = "\\lambda_$(i-1)"

        if p == 0
            push!(terms, λ)
        elseif p == 1
            push!(terms, "$(λ) h")
        else
            push!(terms, "$(λ) h^{$p}")
        end
    end

    return "f(h) = " * join(terms, " + ")
end