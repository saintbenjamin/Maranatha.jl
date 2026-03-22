# ============================================================================
# src/Documentation/Reporter/datapoints/_build_convergence_summary_datapoints_md.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_convergence_summary_datapoints_md(
        a, 
        b, 
        name, 
        hsp, 
        hxp, 
        estp, 
        errp;
        h_power, 
        xscale, 
        yscale, 
        rule, 
        boundary
    ) -> String

Construct a Markdown datapoints-only convergence-summary report.

# Function description

This helper builds a Markdown representation of raw quadrature datapoints for
inspection, documentation, or lightweight archival use.

The generated report contains:

- a run-configuration section,
- a table of filtered step sizes,
- transformed horizontal coordinates,
- quadrature estimates with uncertainties.

Unlike the fit-based reporting helpers, this routine focuses exclusively on the
measured datapoints and plotting convention, without including any extrapolated
value or fit diagnostics.

# Arguments

- `a`, `b`: Integration domain endpoints.

  These may be either scalars (uniform-domain case) or tuples specifying
  per-axis bounds for rectangular domains.

- `name`: Display name of the experiment or dataset.
- `hsp`: Filtered step sizes.
- `hxp`: Filtered transformed step sizes, typically ``h^{p}``.
- `estp`: Filtered quadrature estimates.
- `errp`: Filtered pointwise uncertainties.

# Keyword arguments

- `h_power`: Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale`: Horizontal axis scaling mode.
- `yscale`: Vertical axis scaling mode.
- `rule`: Quadrature rule label.
- `boundary`: Boundary-handling label.

# Returns

- `String`: Markdown-formatted report text.

# Notes

- Numeric formatting relies on the internal helpers
  [`_fmt_md_code_sci`](@ref) and [`_fmt_avgerr_md`](@ref).
- The output is intended to be GitHub-friendly and visually parallel to the
  [``\\LaTeX``](https://www.latex-project.org/) version where possible.
- When any of `a`, `b`, `rule`, or `boundary` is axis-wise, the run
  configuration table expands into one row per axis.
"""
function _build_convergence_summary_datapoints_md(
    a, 
    b, 
    name, 
    hsp,
    hxp, 
    estp, 
    errp;
    h_power,
    xscale,
    yscale,
    rule,
    boundary,
)
    io = IOBuffer()
    cfg_dim = _report_cfg_dim(a, b, rule, boundary)
    plot_setup_txt = "h^$(h_power), $(string(xscale))/$(string(yscale))"

    println(io, "# Convergence datapoints summary: $(name)")
    println(io, "")

    println(io, "## Run configuration")
    println(io, "")

    if cfg_dim == 1
        interval_txt = _fmt_axis_interval_for_run_config(a, b, 1, 1)
        rule_boundary_txt = _fmt_rule_boundary_cell_md(rule, boundary, 1, 1)

        println(io, "| Interval | Rule (Boundary) | Plot setup |")
        println(io, "|:--|:--|:--|")
        println(io,
            "| `$(interval_txt)` | " *
            "`$(rule_boundary_txt)` | " *
            "`$(plot_setup_txt)` |"
        )
    else
        println(io, "| Axis | Rule (Boundary) | Plot setup |")
        println(io, "|:--|:--|:--|")
        for d in 1:cfg_dim
            axis_txt = _fmt_axis_cell_md(a, b, d, cfg_dim)
            rule_boundary_txt = _fmt_rule_boundary_cell_md(rule, boundary, d, cfg_dim)
            println(io,
                "| `$(axis_txt)` | " *
                "`$(rule_boundary_txt)` | " *
                "`$(plot_setup_txt)` |"
            )
        end
    end

    println(io, "")

    println(io, "## Quadrature estimates and uncertainties for different step sizes")
    println(io, "")
    println(io, "| \$h\$ | \$h^$(h_power)\$ | \$I(h)\$ |")
    println(io, "|:--|:--|:--|")

    for i in eachindex(hsp)
        htxt  = _fmt_md_code_sci(hsp[i])
        hptxt = _fmt_md_code_sci(hxp[i])
        qtxt  = _fmt_avgerr_md(estp[i], errp[i])
        println(io, "| $htxt | $hptxt | $qtxt |")
    end

    return String(take!(io))
end
