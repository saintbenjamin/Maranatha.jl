# ============================================================================
# src/Documentation/PlotTools/convergence/plot_convergence_result.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    plot_convergence_result(
        result,
        fit_result;
        name::String = "Maranatha",
        figs_dir::String = ".",
        save_file::Bool = false,
    ) -> Nothing

Plot a fitted convergence study directly from a
[`Maranatha.Runner.run_Maranatha`](@ref) result object and an existing
least-``\\chi^2`` fit result.

# Function description
This routine generates two figures from the supplied run result and fit result:

- an extrapolation panel showing the fitted `I(h)` curve, its uncertainty band,
  the raw datapoints with error bars, and the extrapolated `h = 0` estimate;
- a log-log relative-error panel showing `|I(h)-I_0|/|I_0|`, propagated
  uncertainties, and a unit-slope reference line on the `h^p` axis.

# Arguments
- `result`:
  Result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).

  The plotting routine uses:

  - `result.h` as the scalar step-size sequence,
  - `result.avg` as the measured quadrature estimates,
  - `result.err` as the pointwise error-information objects,
  - `result.rule` and `result.boundary` as labels for output naming.

  In rectangular-domain workflows, `result.h` is expected to be the scalarized
  step-size proxy used for fitting and plotting, rather than the original
  per-axis step tuples.
- `fit_result`:
  Fit result object returned by `least_chi_square_fit`.

  The current implementation expects `fit_result` to provide at least:

  - `params`,
  - `estimate`,
  - `error_estimate`,
  - `cov`,
  - `powers`.

  Optional `chisq` and `dof` fields are also used for plot annotations when
  available.

# Keyword arguments
- `name::String = "Maranatha"`:
  Basename used for figure titles / output filenames.
- `figs_dir::String = "."`:
  Output directory for saved figures.
- `save_file::Bool = false`:
  If `true`, save the generated figures.

# Returns
- `Nothing`:
  This routine is used for its plotting and optional file-output side effects.

# Errors
- Throws an error if input lengths are inconsistent.
- Throws an error if `fit_result` is missing `:powers`.
- Throws an error if `length(fit_result.powers) != length(fit_result.params)`.
- Throws an error if no valid datapoints remain after filtering for either panel.
- Throws an error for the relative-error panel when the extrapolated value is zero.
- Propagates plotting and file-I/O errors.

# Notes
- This function visualizes an existing fit; it does not perform refitting.
- The plotting routine accepts both residual-based and refinement-based
  error-info objects, provided that each entry exposes either `.total` or
  `.estimate`.
- The plotting logic operates on the scalar step-size sequence `result.h`.
- The current horizontal power is inferred from `fit_result.powers[2]`, i.e.
  the first non-constant basis term after the leading constant term.
- Saved filenames encode rule/boundary metadata through
  [`DocUtils._rule_boundary_filename_token`](@ref), so axis-wise specifications
  produce axis-tagged filename tokens.
"""
function plot_convergence_result(
    result,
    fit_result;
    name::String = "Maranatha",
    figs_dir::String = ".",
    save_file::Bool = false,
)

    # ------------------------------------------------------------
    # Direct extraction from run_Maranatha result object
    # ------------------------------------------------------------
    hs = result.h
    estimates = result.avg
    errors = result.err
    rule = result.rule
    boundary = result.boundary

    # ------------------------------------------------------------
    # Determine leading convergence power automatically
    # using composite NC residual model (midpoint expansion)
    # ------------------------------------------------------------

    # --- Input checks ---
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    # Support both residual-based (.total) and refinement-based (.estimate) error objects.
    @inline function _extract_error_total(e)
        if hasproperty(e, :total)
            return float(e.total)
        elseif hasproperty(e, :estimate)
            return abs(float(e.estimate))
        else
            JobLoggerTools.error_benji(
                "Unsupported error-info structure for plotting (need :total or :estimate)."
            )
        end
    end

    # ------------------------------------------------------------
    # Determine x-axis power from fit (e.g. h^p)
    # ------------------------------------------------------------
    fit_powers = if hasproperty(fit_result, :powers)
        fit_result.powers
    else
        JobLoggerTools.error_benji(
            "fit_result missing :powers (cannot infer convergence power)"
        )
    end

    # first nonzero power
    lead_pow = fit_powers[2]   # index 1 is 0 (constant term)

    # x-axis = h^lead_pow
    hx = hs .^ lead_pow

    errors_val = [_extract_error_total(e) for e in errors]

    errors_pos = abs.(errors_val)

    mask = (hx .> 0) .& isfinite.(hx) .& isfinite.(estimates) .& isfinite.(errors_pos)

    hxp = hx[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    isempty(hxp) && JobLoggerTools.error_benji(
        "No valid points to plot."
    )

    # --- New fit result structure ---
    pvec = fit_result.params
    I0      = fit_result.estimate
    I0_err  = fit_result.error_estimate

    # --- Build model automatically from params ---
    Cov = fit_result.cov

    # enforce symmetry for numerical stability
    CovS = LinearAlgebra.Symmetric(Matrix(Cov))

    # --------------------------------------------
    # Determine model exponents used by the fit
    # Prefer fit_result.powers if present; otherwise fall back.
    # --------------------------------------------
    fit_powers = if hasproperty(fit_result, :powers)
        fit_result.powers
    end

    (length(fit_powers) == length(pvec)) || JobLoggerTools.error_benji(
        "fit_result.powers length mismatch: expected $(length(pvec)), got $(length(fit_powers))"
    )

    function basis_vec(h)
        T = typeof(h)
        v = Vector{T}(undef, length(pvec))
        @inbounds for i in 1:length(pvec)
            pow = fit_powers[i]
            v[i] = (pow == 0) ? one(T) : h^pow
        end
        return v
    end

    function model_and_err(h)
        φ = basis_vec(h)
        y = LinearAlgebra.dot(pvec, φ)

        var = LinearAlgebra.dot(φ, CovS * φ)
        σ = sqrt(abs(var))
        return y, σ
    end

    # --- Smooth curve including extrapolated point at x = 0 ---
    xmin = minimum(hxp)
    xmax = maximum(hxp)

    x_range_log = 10 .^ range(log10(xmin), log10(xmax); length=200)

    # prepend zero explicitly
    x_range = vcat(zero(eltype(x_range_log)), x_range_log)

    # model needs h, not x; x = h^lead_pow  =>  h = x^(1/lead_pow)
    h_range = x_range .^ (one(eltype(x_range)) / convert(eltype(x_range), lead_pow))

    y_fit = similar(h_range)
    y_err = similar(h_range)

    for i in eachindex(h_range)
        y_fit[i], y_err[i] = model_and_err(h_range[i])
    end

    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    # Fit curve
    ax.plot(
        x_range, 
        y_fit; 
        color="black", 
        linewidth=2.5
    )

    # --- Fit error band ---
    ax.fill_between(
        x_range,
        y_fit .- y_err,
        y_fit .+ y_err;
        alpha=0.25,
        linewidth=0,
        color="black"
    )

    # Data points
    ax.errorbar(
        hxp, 
        estp;
        yerr=errp,
        fmt="o",
        color="blue",
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="blue"
    )

    # --- Extrapolated point at h = 0 ---
    ax.errorbar(
        [0.0],
        [I0];
        yerr=[I0_err],
        fmt="s",
        color="red",
        markersize=8,
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="red"
    )

    ax.set_xlabel("\$h^{ $(round(Int, lead_pow)) }\$")
    ax.set_ylabel("\$I(h)\$")

    # ---- annotate plot 1 (I0 +/- err, chi2/dof) ----
    chisq = hasproperty(fit_result, :chisq) ? fit_result.chisq : NaN
    dof   = hasproperty(fit_result, :dof)   ? fit_result.dof   : NaN
    red   = (isfinite(chisq) && isfinite(dof) && dof != 0) ? chisq / dof : NaN

    txt_I0 = try
        I0_e2d = AvgErrFormatter.avgerr_e2d_from_float(I0, I0_err; latex_grouping=true)
        "\$I_0 = $I0_e2d\$"
    catch
        @sprintf("\$I_0 = %.7g \\pm %.7g\$", I0, I0_err)
    end

    txt1 = txt_I0 * "\n" * @sprintf("\$\\chi^2/\\mathrm{d.o.f.} = \\texttt{%.7g}\$", red)

    _smart_text_placement!(
        fig, 
        ax;
        text=txt1,
        x_points=collect(hxp),
        y_points=collect(estp),
        yerr_points=collect(errp),
        x_curve=collect(x_range),
        y_curve=collect(y_fit),
        fontsize=11
    )

    display(fig)

    spec_str = DocUtils._rule_boundary_filename_token(result.a, result.b, rule, boundary)

    basename = "result_$(name)_$(spec_str)_extrap"
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

    # ============================================================
    # Extra plot: log-log convergence of relative error
    #   y = |I(h) - I0| / |I0|
    # with error bars for both data points and fit curve
    # outfile: ..._rel_log.png
    # ============================================================

    absI0 = abs(I0)
    (absI0 > 0) || JobLoggerTools.error_benji(
        "Relative-error plot requires nonzero I0 (got I0=$I0)."
    )

    # --- Data for relative-error plot ---
    Δp   = estp .- I0
    relp = abs.(Δp) ./ absI0

    rel_errp = sqrt.( (errp ./ absI0).^2 .+ ((abs.(Δp) .* I0_err) ./ (absI0^2)).^2 )

    mask2 = (relp .> 0) .& isfinite.(relp) .& isfinite.(rel_errp) .& (rel_errp .> 0) .&
            (hxp .> 0) .& isfinite.(hxp)

    hxp2 = hxp[mask2]
    rel2 = relp[mask2]
    rerr2 = rel_errp[mask2]

    isempty(hxp2) && JobLoggerTools.error_benji(
        "No valid positive relative-error points for log-log plot."
    )

    # --- Fit curve + curve uncertainty band for relative error ---
    x_range2 = x_range_log
    h_range2 = x_range2 .^ (one(eltype(x_range2)) / convert(eltype(x_range2), lead_pow))

    rel_fit = similar(h_range2)
    rel_sig = similar(h_range2)

    for i in eachindex(h_range2)
        yh, σy = model_and_err(h_range2[i])
        Δ = yh - I0

        rel_fit[i] = abs(Δ) / absI0
        rel_sig[i] = sqrt( (σy / absI0)^2 + ((abs(Δ) * I0_err) / (absI0^2))^2 )
    end

    # --- Plot ---
    fig2, ax2 = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    ax2.plot(x_range2, rel_fit; color="red", linewidth=2.5)

    # --- Reference slope line (slope = 1 in h^p axis) ---
    idx_ref = length(x_range2) ÷ 2
    x_ref = x_range2[idx_ref]
    y_ref = rel_fit[idx_ref]

    ref_line = y_ref .* (x_range2 ./ x_ref)

    ax2.plot(
        x_range2,
        ref_line;
        linestyle="--",
        linewidth=2.0,
        color="gray"
    )

    # Fit curve error band
    ax2.fill_between(
        x_range2,
        rel_fit .- rel_sig,
        rel_fit .+ rel_sig;
        alpha=0.25,
        linewidth=0,
        color="red"
    )

    # Data points with error bars
    ax2.errorbar(
        hxp2, 
        rel2;
        yerr=rerr2,
        fmt="o",
        color="blue",
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="blue"
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_xlabel("\$h^{ $(round(Int, lead_pow)) }\$")
    ax2.set_ylabel(raw"$|I(h)-I_0|/|I_0|$")

    txt_I0 = try
        I0_e2d = AvgErrFormatter.avgerr_e2d_from_float(
            I0, 
            I0_err; 
            latex_grouping=true
        )
        "\$I_0 = $I0_e2d\$"
    catch
        @sprintf("\$I_0 = %.7g \\pm %.7g\$", I0, I0_err)
    end

    txt2 = txt_I0 * "\n" * @sprintf("\$\\chi^2/\\mathrm{d.o.f.} = \\texttt{%.7g}\$", red)

    _smart_text_placement!(fig2, ax2;
        text=txt2,
        x_points=collect(hxp2),
        y_points=collect(rel2),
        yerr_points=collect(rerr2),
        x_curve=collect(x_range2),
        y_curve=collect(rel_fit),
        fontsize=11
    )

    display(fig2)

    basename = "result_$(name)_$(spec_str)_reldiff"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    fig2.tight_layout()
    PyPlot.close(fig2)

    return nothing
end
