# ============================================================================
# src/Documentation/PlotTools/datapoints/plot_datapoints_result.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    plot_datapoints_result(
        result;
        name::String = "Maranatha",
        h_power::Real = 1,
        xscale::Symbol = :linear,
        yscale::Symbol = :linear,
        figs_dir::String = ".",
        save_file::Bool = false,
    ) -> Nothing

Plot raw convergence datapoints directly from a
[`Maranatha.Runner.run_Maranatha`](@ref) result object, without a fitted model,
for pre-fit inspection of alignment, scaling, and possible oscillatory behavior.

# Arguments
- `result`:
  Result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).

  The plotting routine uses:

  - `result.h` as the scalar step-size sequence,
  - `result.avg` as the raw quadrature estimates,
  - `result.err` as the error-information objects used for plotting error bars,
  - `result.rule` and `result.boundary` as labels for output naming.

  In rectangular-domain workflows, `result.h` is expected to be the scalarized
  step-size proxy stored for downstream plotting, rather than the original
  per-axis step tuples.

# Keyword arguments
- `name::String = "Maranatha"`:
  Basename used for output filenames.
- `h_power::Real = 1`:
  Power `p` used to construct the horizontal coordinate `x = h^p`.
- `xscale::Symbol = :linear`, `yscale::Symbol = :linear`:
  Axis scaling options. Supported values are `:linear` and `:log`.
- `figs_dir::String = "."`:
  Output directory for saved figures.
- `save_file::Bool = false`:
  If `true`, save the generated figure.

# Returns
- `Nothing`:
  This routine is used for plotting and optional file-output side effects.

# Errors
- Throws an error if input lengths are inconsistent.
- Throws an error if unsupported axis-scale keywords are provided.
- Throws an error if no valid datapoints remain after filtering.
- Propagates plotting and file-I/O errors.

# Notes
- This routine is intended as a diagnostic plotter before fitting.
- This routine accepts both residual-based and refinement-based error-info
  objects, provided that each entry exposes either `.total` or `.estimate`.
- The plotting logic operates on the scalar step-size sequence `result.h`.
- Saved filenames encode rule/boundary metadata through
  [`DocUtils._rule_boundary_filename_token`](@ref), so axis-wise specifications
  produce axis-tagged filename tokens.
"""
function plot_datapoints_result(
    result;
    name::String = "Maranatha",
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    figs_dir::String = ".",
    save_file::Bool = false,
)
    hs = result.h
    estimates = result.avg
    errors = result.err
    rule = result.rule
    boundary = result.boundary

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

    @inline function _extract_error_total(e)
        if hasproperty(e, :total)
            return e.total
        elseif hasproperty(e, :estimate)
            return abs(e.estimate)
        else
            JobLoggerTools.error_benji(
                "Unsupported error-info structure for datapoint plot (need :total or :estimate)."
            )
        end
    end

    T = promote_type(eltype(hs), eltype(estimates), typeof(h_power))

    xvals_raw = (T.(hs)) .^ convert(T, h_power)
    yvals_raw = T.(estimates)
    yerrs_raw = T.([_extract_error_total(e) for e in errors])

    mask = isfinite.(Float64.(xvals_raw)) .&
           isfinite.(Float64.(yvals_raw)) .&
           isfinite.(Float64.(yerrs_raw))

    if xscale == :log
        mask .&= xvals_raw .> zero(T)
    end
    if yscale == :log
        mask .&= yvals_raw .> zero(T)
    end

    xp = Float64.(xvals_raw[mask])
    yp = Float64.(yvals_raw[mask])
    ep = Float64.(yerrs_raw[mask])

    isempty(xp) && JobLoggerTools.error_benji(
        "No valid datapoints remain after filtering for plot_datapoints_result."
    )

    p = sortperm(xp; rev = true)
    xp = xp[p]
    yp = yp[p]
    ep = ep[p]

    ylabel_txt = raw"$I(h)$"

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

    display_name, file_name = DocUtils._split_report_name(name)

    spec_str = DocUtils._rule_boundary_filename_token(result.a, result.b, rule, boundary)

    basename = "$(file_name)_$(spec_str)_datapoints_hpow_$(h_power)_$(xscale)_$(yscale)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")

    mkpath(figs_dir)
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force = true)
        end
    end

    fig.tight_layout()
    PyPlot.close(fig)

    return nothing
end
