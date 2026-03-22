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
        a, 
        b, 
        name, 
        hsp, 
        hxp, 
        estp, 
        errp,
        pvec, 
        λerr, 
        fit_powers, 
        I0, 
        I0_err, 
        red, 
        nerr_terms;
        rule, 
        boundary
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

- `a`, `b`: Integration domain endpoints.

  These may be either scalars (uniform-domain case) or tuples specifying
  per-axis bounds for rectangular domains.

- `name`: Display name of the experiment or dataset.
- `hsp`: Filtered scalar step sizes used in the fit / report.
- `hxp`: Filtered transformed horizontal coordinates, typically ``h^{p}``.
- `estp`: Filtered quadrature estimates.
- `errp`: Filtered pointwise uncertainties.
- `pvec`: Best-fit parameter vector.
- `λerr`: One-sigma parameter uncertainties.
- `fit_powers`: Powers used in the fitted model basis.
- `I0`: Extrapolated ``h \\to 0`` estimate.
- `I0_err`: Uncertainty of the extrapolated estimate.
- `red`: Reduced chi-square value.
- `nerr_terms`: Number of error terms used in the derivative-based model,
  when applicable.

# Returns

- `String`: Markdown report text.

# Notes

- Designed for GitHub-compatible Markdown rendering.
- Mathematical expressions are emitted using fenced `math` blocks.
- When any of `a`, `b`, `rule`, or `boundary` is axis-wise, the run
  configuration table expands into one row per axis.
- The tabulated `h` columns always use the scalarized step sequence supplied to
  the fitter/reporting pipeline.
"""
function _build_convergence_summary_md(
    a, 
    b, 
    name, 
    hsp, 
    hxp, 
    estp, 
    errp,
    pvec, 
    λerr, 
    fit_powers,
    I0, 
    I0_err, 
    red, 
    nerr_terms;
    rule, 
    boundary, 
    err_method=:refinement
)
    io = IOBuffer()
    nerr_terms_eff = (err_method == :refinement) ? 0 : nerr_terms
    cfg_dim = _report_cfg_dim(a, b, rule, boundary)

    println(io, "# Convergence summary: $(name)")
    println(io, "")

    println(io, "## Run configuration")
    println(io, "")

    if cfg_dim == 1
        interval_txt = _fmt_axis_interval_for_run_config(a, b, 1, 1)
        rule_boundary_txt = _fmt_rule_boundary_cell_md(rule, boundary, 1, 1)

        if nerr_terms_eff == 0
            println(io, "| Interval | Rule (Boundary) |")
            println(io, "|:--|:--|")
            println(io, "| `$(interval_txt)` | `$(rule_boundary_txt)` |")
        else
            println(io, "| Interval | Rule (Boundary) | Number of error terms |")
            println(io, "|:--|:--|:--|")
            println(io, "| `$(interval_txt)` | `$(rule_boundary_txt)` | `$(nerr_terms_eff)` |")
        end
    else
        if nerr_terms_eff == 0
            println(io, "| Axis | Rule (Boundary) |")
            println(io, "|:--|:--|")
            for d in 1:cfg_dim
                axis_txt = _fmt_axis_cell_md(a, b, d, cfg_dim)
                rule_boundary_txt = _fmt_rule_boundary_cell_md(rule, boundary, d, cfg_dim)
                println(io, "| `$(axis_txt)` | `$(rule_boundary_txt)` |")
            end
        else
            println(io, "| Axis | Rule (Boundary) | Number of error terms |")
            println(io, "|:--|:--|:--|")
            for d in 1:cfg_dim
                axis_txt = _fmt_axis_cell_md(a, b, d, cfg_dim)
                rule_boundary_txt = _fmt_rule_boundary_cell_md(rule, boundary, d, cfg_dim)
                println(io, "| `$(axis_txt)` | `$(rule_boundary_txt)` | `$(nerr_terms_eff)` |")
            end
        end
    end

    println(io, "")

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

    println(io, "| $htxt | $hptxt | **$qtxt** |")
    println(io, "")

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

        if fit_powers[i] == 0
            println(io, "| $λname | **$λtxt** |")
        else
            println(io, "| $λname | $λtxt |")
        end
    end

    println(io, "| \$\\chi^2 / \\text{d.o.f.}\$ | $(_fmt_md_code_sci(red)) |")

    return String(take!(io))
end
