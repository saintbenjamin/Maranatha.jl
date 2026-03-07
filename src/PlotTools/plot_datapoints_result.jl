"""
    plot_datapoints_result(
        name::String,
        hs::Vector{Float64},
        estimates::Vector{Float64},
        errors::Vector;
        h_power::Real = 1,
        xscale::Symbol = :linear,
        yscale::Symbol = :linear,
        ymode::Symbol = :value,
        reference_value = nothing,
        rule::Symbol = :gauss_p3,
        boundary::Symbol = :LU_ININ,
        figs_dir::String = ".",
        save_file::Bool = false
    ) -> Nothing

Plot raw datapoints only, without any fitted model, in order to inspect
alignment, oscillation, and scaling behavior before performing a fit.

# Function description

This routine is intended as a pre-fit diagnostic visualization tool.
It plots only the sampled datapoints (and optional error bars), allowing
the user to inspect whether the convergence data appears smooth, roughly
linear in a chosen transformed coordinate, or contaminated by oscillatory
behavior across resolutions.

The horizontal axis is constructed manually as
```math
x = h^{p},
```
where `p = h_power` is chosen by the caller.

This makes it possible to test empirically which power of `h`
best reveals approximately linear behavior before attempting a
least-`\\chi^2` extrapolation.

Optional logarithmic scaling can be enabled independently on both axes.

# Arguments

`name`
: Label used in the output filename.

`hs`
: Step sizes `h`.

`estimates`
: Quadrature estimates corresponding to `hs`.

`errors`
: Collection of error-information objects.  Each entry is expected
to provide a `.total` field.

# Keyword arguments

`h_power`
: Power `p` used to construct the horizontal coordinate `x = h^p`.

`xscale`
: Axis scale for `x`. Supported values are `:linear` and `:log`.

`yscale`
: Axis scale for `y`. Supported values are `:linear` and `:log`.

`rule`, `boundary`
: Labels used in output filenames.

`figs_dir`
: Output directory for saved figures.

`save_file`
: If `true`, save the plot as a PDF.

# Returns

`nothing`
"""
function plot_datapoints_result(
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector;
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    rule::Symbol = :gauss_p3,
    boundary::Symbol = :LU_ININ,
    figs_dir::String = ".",
    save_file::Bool = false,
)
    n = length(hs)
    length(estimates) == n || JobLoggerTools.error_benji(
        "Input length mismatch: length(estimates) != length(hs)"
    )
    length(errors) == n || JobLoggerTools.error_benji(
        "Input length mismatch: length(errors) != length(hs)"
    )

    (xscale == :linear || xscale == :log) || JobLoggerTools.error_benji(
        "Unsupported xscale=$xscale (expected :linear or :log)"
    )
    (yscale == :linear || yscale == :log) || JobLoggerTools.error_benji(
        "Unsupported yscale=$yscale (expected :linear or :log)"
    )

    xvals = Float64.(hs) .^ float(h_power)
    err_abs = abs.([e.total for e in errors])

    yvals = Float64[]
    yerrs = Float64[]
    ylabel_txt = ""

    yvals = Float64.(estimates)
    yerrs = err_abs
    ylabel_txt = raw"$I(h)$"

    mask = isfinite.(xvals) .& isfinite.(yvals) .& isfinite.(yerrs)

    if xscale == :log
        mask .&= xvals .> 0
    end
    if yscale == :log
        mask .&= yvals .> 0
    end

    xp = xvals[mask]
    yp = yvals[mask]
    ep = yerrs[mask]

    isempty(xp) && JobLoggerTools.error_benji(
        "No valid datapoints remain after filtering for plot_datapoints_result."
    )

    p = sortperm(xp; rev=true)
    xp = xp[p]
    yp = yp[p]
    ep = ep[p]

    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6, 5.0), dpi=500)

    ax.errorbar(
        xp, yp;
        yerr = ep,
        fmt = "o",
        color = "blue",
        capsize = 6,
        markerfacecolor = "none",
        markeredgecolor = "blue"
    )

    if xscale == :log
        ax.set_xscale("log")
    end
    if yscale == :log
        ax.set_yscale("log")
    end

    ax.set_xlabel("\$h^{$(h_power)}\$")
    ax.set_ylabel(ylabel_txt)

    txt = "xscale = $(xscale), yscale = $(yscale)\n" *
          "power = $(h_power)"

    _smart_text_placement!(fig, ax;
        text = txt,
        x_points = collect(xp),
        y_points = collect(yp),
        yerr_points = collect(ep),
        fontsize = 11
    )

    display(fig)

    basename = "result_$(name)_$(String(rule))_$(String(boundary))_datapoints_hpow_$(h_power)_$(xscale)_$(yscale)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")

    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    fig.tight_layout()
    PyPlot.close(fig)

    return nothing
end