# ============================================================================
# src/Documentation/Reporter/_build_convergence_summary_md.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_convergence_summary_md(
        a, b, name, hsp, hxp, estp, errp,
        pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
        rule, boundary
    ) -> String

Construct a Markdown convergence-summary report.

# Function description

This helper produces a Markdown-formatted report containing:

- run configuration,
- quadrature estimates across resolutions,
- extrapolated value,
- fit results,
- explicit model expression in a math block.

The structure mirrors the [``\\LaTeX``](https://www.latex-project.org/) version for consistency.

# Arguments

Same as [`_build_convergence_summary_tex`](@ref).

# Returns

- `String`: Markdown report text.

# Notes

- Designed for GitHub-compatible Markdown rendering.
- Mathematical expressions are emitted using fenced `math` blocks.
"""
function _build_convergence_summary_md(
    a, b, name, hsp, hxp, estp, errp,
    pvec, λerr, fit_powers, I0, I0_err, red, nerr_terms;
    rule, boundary
)
    io = IOBuffer()

    println(io, "# Convergence summary: $(name)")
    println(io, "")

    # ------------------------------------------------------------
    # Run configuration (column style — matches LaTeX)
    # ------------------------------------------------------------
    println(io, "## Run configuration")
    println(io, "")
    println(io, "| Interval | Rule (Boundary) | Number of error terms |")
    println(io, "|:--|:--|:--|")
    println(io,
        "| `[$(a), $(b)]` | " *
        "`$(String(rule)) ($(String(boundary)))` | " *
        "`$(nerr_terms)` |"
    )
    println(io, "")

    # ------------------------------------------------------------
    # Quadrature estimates
    # ------------------------------------------------------------
    println(io, "## Quadrature estimates and uncertainties for different step sizes")
    println(io, "")
    println(io, "| \$h\$ | \$h^$(fit_powers[2])\$ | \$I(h)\$ |")
    println(io, "|:--|:--|:--|")

    for i in eachindex(hsp)
        htxt  = _fmt_md_code_sci(hsp[i])
        hptxt = _fmt_md_code_sci(hxp[i])
        qtxt  = _fmt_avgerr_md(estp[i], errp[i])
        println(io, "| $htxt | $hptxt | $qtxt |")
    end

    htxt  = _fmt_md_code_sci(0.0)
    hptxt = _fmt_md_code_sci(0.0)
    qtxt  = _fmt_avgerr_md(I0, I0_err)

    # highlight extrapolated value (Markdown version of bold)
    println(io, "| $htxt | $hptxt | **$qtxt** |")
    println(io, "")

    # ------------------------------------------------------------
    # Fit results
    # ------------------------------------------------------------
    println(io, "## Least-chi-square fit results for extrapolation to \$h \\to 0\$")
    println(io, "")
    fit_model_tex = _build_fit_model_tex(fit_powers)
    println(io, "```math")
    println(io, fit_model_tex)
    println(io, "```")
    println(io, "")
    println(io, "| parameter | fit result |")
    println(io, "|:--|:--|")

    for i in eachindex(pvec)
        λname = fit_powers[i] == 0 ? "\$\\lambda_0\$" : "\$\\lambda_$(i-1)\$"
        λtxt  = _fmt_avgerr_md(pvec[i], λerr[i])

        # bold constant term to match LaTeX
        if fit_powers[i] == 0
            println(io, "| $λname | **$λtxt** |")
        else
            println(io, "| $λname | $λtxt |")
        end
    end

    println(io, "| \$\\chi^2 / \\text{d.o.f.}\$ | $(_fmt_md_code_sci(red)) |")

    return String(take!(io))
end