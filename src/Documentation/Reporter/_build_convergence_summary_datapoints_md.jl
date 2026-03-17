# ============================================================================
# src/Documentation/Reporter/_build_convergence_summary_datapoints_md.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_convergence_summary_datapoints_md(
        a, b, name, hsp, hxp, estp, errp;
        h_power, xscale, yscale, rule, boundary
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

- `a`, `b`: Integration interval endpoints.
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
"""
function _build_convergence_summary_datapoints_md(
    a, b, name, hsp, hxp, estp, errp;
    h_power,
    xscale,
    yscale,
    rule,
    boundary,
)
    io = IOBuffer()

    println(io, "# Convergence datapoints summary: $(name)")
    println(io, "")

    println(io, "## Run configuration")
    println(io, "")
    println(io, "| Interval | Rule (Boundary) | Plot setup |")
    println(io, "|:--|:--|:--|")
    println(io,
        "| `[$(a), $(b)]` | " *
        "`$(String(rule)) ($(String(boundary)))` | " *
        "`h^$(h_power)`, `$(String(xscale))/$(String(yscale))` |"
    )
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