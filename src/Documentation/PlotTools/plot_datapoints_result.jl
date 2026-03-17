# ============================================================================
# src/Documentation/PlotTools/plot_datapoint_result.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _split_report_name(
        name::AbstractString
    ) -> Tuple{String, String}

Split a user-facing report name into a display name and a filesystem-safe base
name.

# Function description
This helper normalizes a report identifier for two separate uses:

- `display_name`: the original input converted to `String`, preserved for
  human-facing display contexts, and
- `file_name`: the basename of the input path with any trailing `.jld2`
  suffix removed, intended for output filename construction.

This is useful when plotting or reporting functions accept either a plain name
or a path-like string and need a clean filename stem for saved artifacts.

# Arguments
- `name::AbstractString`:
  Report name or path-like string.

# Returns
- `Tuple{String, String}`:
  A pair `(display_name, file_name)`.

# Notes
- `display_name` preserves the full input string content.
- `file_name` is derived from `basename(String(name))`.
- Only a trailing `.jld2` suffix is stripped.
"""
function _split_report_name(name::AbstractString)
    display_name = String(name)
    file_name = replace(basename(String(name)), r"\.jld2$" => "")
    return display_name, file_name
end

"""
    plot_datapoints_result(
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
    ) -> Nothing

Plot raw convergence datapoints without a fitted model, for pre-fit inspection of
alignment, scaling, and possible oscillatory behavior.

# Arguments
- `name::String`:
  Basename used for output filenames.
- `hs::Vector{Float64}`:
  Step sizes used to define the horizontal coordinate.
- `estimates::Vector{Float64}`:
  Raw quadrature estimates.
- `errors::Vector`:
  Collection of error-information objects used for plotting error bars.

  Each entry is expected to provide either:

  - a `.total` field, as in the residual-based error-estimation workflow, or
  - an `.estimate` field, as in the refinement-based error-estimation workflow.

  The plotting routine converts each entry into a nonnegative scalar plotting
  uncertainty through an internal extractor.

# Keyword arguments
- `h_power::Real = 1`:
  Power `p` used to construct the horizontal coordinate `x = h^p`.
- `xscale::Symbol = :linear`, `yscale::Symbol = :linear`:
  Axis scaling options. Supported values are `:linear` and `:log`.
- `rule::Symbol = :gauss_p3`, `boundary::Symbol = :LU_ININ`:
  Labels used in output filenames.
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
- A convenience wrapper `plot_datapoints_result(result; ...)` is also provided.
- This routine accepts both residual-based and refinement-based error-info
  objects, provided that each entry exposes either `.total` or `.estimate`.
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

    # Support both residual-based (.total) and refinement-based (.estimate) error objects.
    @inline function _extract_error_total(e)
        if hasproperty(e, :total)
            return float(e.total)
        elseif hasproperty(e, :estimate)
            return abs(float(e.estimate))
        else
            JobLoggerTools.error_benji(
                "Unsupported error-info structure for datapoint plot (need :total or :estimate)."
            )
        end
    end

    xvals = Float64.(hs) .^ float(h_power)
    err_abs = [_extract_error_total(e) for e in errors]

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

    display_name, file_name = _split_report_name(name)

    basename = "$(file_name)_$(String(rule))_$(String(boundary))_datapoints_hpow_$(h_power)_$(xscale)_$(yscale)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")

    mkpath(figs_dir)
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

"""
    plot_datapoints_result(
        result;
        name::String = "Maranatha",
        h_power::Real = 1,
        xscale::Symbol = :linear,
        yscale::Symbol = :linear,
        rule::Symbol = result.rule,
        boundary::Symbol = result.boundary,
        figs_dir::String = ".",
        save_file::Bool = false,
    ) -> Nothing

Convenience wrapper that extracts raw datapoints from a Maranatha run result and
forwards them to the primary `plot_datapoints_result` method.

# Arguments
- `result`:
  Result object returned by `run_Maranatha`.

# Keyword arguments
- `name::String = "Maranatha"`:
  Basename used for output filenames.
- `h_power::Real = 1`:
  Power used to construct the horizontal coordinate.
- `xscale::Symbol = :linear`, `yscale::Symbol = :linear`:
  Axis scaling options forwarded to the primary method.
- `rule::Symbol = result.rule`, `boundary::Symbol = result.boundary`:
  Labels forwarded to the primary method.
- `figs_dir::String = "."`:
  Output directory for saved figures.
- `save_file::Bool = false`:
  If `true`, save the generated figure.

# Returns
- `Nothing`.

# Errors
- Propagates all validation and plotting errors from the primary method.
"""
function plot_datapoints_result(
    result;
    name::String = "Maranatha",
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    figs_dir::String = ".",
    save_file::Bool = false,
)
    return plot_datapoints_result(
        name,
        Vector{Float64}(result.h),
        Vector{Float64}(result.avg),
        result.err;
        h_power = h_power,
        xscale = xscale,
        yscale = yscale,
        rule = rule,
        boundary = boundary,
        figs_dir = figs_dir,
        save_file = save_file,
    )
end