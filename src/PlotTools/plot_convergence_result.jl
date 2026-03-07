"""
    plot_convergence_result(
        a::Real,
        b::Real,
        name::String,
        hs::Vector{Float64},
        estimates::Vector{Float64},
        errors::Vector,
        fit_result;
        rule::Symbol = :gauss_p3,
        boundary::Symbol = :LU_ININ,
        figs_dir::String = ".",
        save_file::Bool = false
    ) -> Nothing

Plot a fitted convergence study by showing ``I(h)`` against ``h^{p}``, overlaying the
reconstructed fit curve and its propagated uncertainty band, and generating an additional
log-log plot of the relative extrapolation error.

This is typically the **third step** in a standard `Maranatha.jl` workflow:
first generate `result` with [`Maranatha.Runner.run_Maranatha`](@ref),
then fit with [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref),
and finally visualize the result with `plot_convergence_result`.

# Function description
This routine is a visualization companion to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

It is designed to work with the same raw convergence-data structure returned by
[`Maranatha.Runner.run_Maranatha`](@ref), together with a downstream fit result returned by
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

In a typical workflow, the inputs are passed almost directly from those earlier stages:
`hs = result.h`, `estimates = result.avg`, `errors = result.err`, and
`fit_result = fit`.

In particular, the `errors` input is expected to be a collection of error-information
objects where each entry provides a `.total` field used as the plotted pointwise
error-bar magnitude.

This routine reconstructs the fitted convergence model directly from the stored fit result
without performing any new fit, and produces two figures:

1. a main convergence plot of ``I(h)`` versus ``h^{p}``
2. a log-log relative-error plot of ``\\dfrac{|I(h)-I_0|}{|I_0|}`` versus ``h^{p}``

Here ``p`` is taken as the first non-constant power stored in `fit_result.powers`,
namely `fit_result.powers[2]`, so the horizontal axis reflects the leading fitted
convergence scale rather than the raw step size ``h`` itself.

The plotted ``x`` coordinate is therefore

```math
x = h^{p},
```

while the fitted model itself is still evaluated as a function of `h`.
Internally, a dense grid is constructed in ``x``, converted back to ``h`` via

```math
h = x^{1/p},
```

and then used to evaluate the reconstructed model and its propagated uncertainty.

## Model reconstruction (no refitting)

This function does *not* refit the data.
Instead, it uses the already-computed fit result to reconstruct the model and its
uncertainty directly from:

* `fit_result.params`
* `fit_result.cov`
* `fit_result.estimate`
* `fit_result.error_estimate`
* `fit_result.powers`

This makes the plotting stage reproducible and consistent with the actual fitted basis,
including any forward-shift (`ff_shift`) that may have been applied during fitting.

The basis is reconstructed using the stored exponent vector:

```math
I(h) = \\bm{\\lambda}^{\\mathsf{T}} \\varphi(h),
\\qquad
\\varphi_1(h)=1,
\\qquad
\\varphi_i(h)=h^{\\texttt{powers[i]}} \\quad (i \\ge 2),
```

with `powers = fit_result.powers` and `length(powers) == length(params)` required.

## Main convergence plot

The main figure contains:

* the reconstructed fit curve ``I_{\\mathrm{fit}}(h)``
* a ``1\\,\\sigma`` fit band propagated from the parameter covariance
* measured quadrature estimates with pointwise error bars
* the extrapolated value at ``h^{p}=0`` with uncertainty `fit_result.error_estimate`

The fit uncertainty is propagated as

```math
\\sigma_{\\mathrm{fit}}(h)^2 = \\varphi(h)^{\\mathsf{T}} V \\varphi(h),
```

where ``V = \\texttt{fit_result.cov}``.

A smart text-placement helper is used to position the annotation box
(``I_0`` and ``\\chi^2/\\mathrm{d.o.f.}``) while heuristically avoiding data points,
error bars, and the fitted curve.

## Relative-error log-log plot

A second figure is produced showing the relative convergence error

```math
r(h) = \\frac{|I(h)-I_0|}{|I_0|},
\\qquad x = h^{p},
```

on log-log axes.

The corresponding fitted relative-error curve is

```math
r_{\\mathrm{fit}}(h) = \\frac{|I_{\\mathrm{fit}}(h)-I_0|}{|I_0|}.
```

### Error bars for measured points

For each measured point, the relative-error uncertainty is propagated to first order,
assuming independent uncertainties for ``I(h)`` and ``I_0``:

```math
\\sigma_r^2 \\approx
\\left(\\frac{\\sigma_I}{|I_0|}\\right)^2
+
\\left(\\frac{|I(h)-I_0|}{|I_0|^2}\\,\\sigma_{I_0}\\right)^2.
```

### Uncertainty band for the fitted curve

The relative-error fit band is propagated as

```math
\\sigma_{r,\\mathrm{fit}}^2(h) \\approx
\\left(\\frac{\\sigma_{\\mathrm{fit}}(h)}{|I_0|}\\right)^2
+
\\left(\\frac{|I_{\\mathrm{fit}}(h)-I_0|}{|I_0|^2}\\,\\sigma_{I_0}\\right)^2.
```

A dashed slope-1 reference line is also drawn in the ``x = h^{p}`` coordinate,
corresponding to the expected leading-order behavior ``r \\propto h^{p}``.

As in the main plot, the annotation box is placed automatically using the same
smart overlap-avoidance helper.

## Output files

When `save_file=true`, two PDF files are written under `figs_dir`:

```julia
result_\$(name)_\$(rule)_\$(boundary)_extrap.pdf
result_\$(name)_\$(rule)_\$(boundary)_reldiff.pdf
```

If the external command `pdfcrop` is available, each saved PDF is cropped automatically.

# Arguments

* `a`, `b` :
  Integration bounds. These are retained for API consistency, although the plotting
  routine itself mainly uses the supplied `hs`, `estimates`, and `errors`.
* `name` :
  Label used in the output filenames.
* `hs` :
  Step sizes ``h``.
* `estimates` :
  Quadrature estimates ``I(h)`` corresponding to `hs`.
* `errors` :
  Collection of error-information objects associated with the sampled estimates.
  In the current implementation, each entry is expected to provide a `.total` field
  (for example, the objects stored in `result.err` returned by
  [`Maranatha.Runner.run_Maranatha`](@ref)).
  Absolute values of these `.total` entries are used for plotting the pointwise error bars.
* `fit_result` :
  Fit object, typically the `NamedTuple` returned by
  [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

  It is expected to provide at least:

  * `fit_result.params`
  * `fit_result.cov`
  * `fit_result.estimate`
  * `fit_result.error_estimate`
  * `fit_result.powers`

# Keyword arguments

* `rule::Symbol=:gauss_p3` :
  Rule label used in output filenames.
* `boundary::Symbol=:LU_ININ` :
  Boundary-condition label used in output filenames.
* `figs_dir::String="."` :
  Directory in which output PDFs are saved when `save_file=true`.
* `save_file::Bool=false` :
  If `true`, save the generated figures as PDF files.

# Typical workflow context

A common usage pattern is:

1. generate `result` with [`Maranatha.Runner.run_Maranatha`](@ref)
2. generate `fit` with [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
3. call [`plot_convergence_result`](@ref) using
   `result.h`, `result.avg`, `result.err`, and `fit`

# Returns

`nothing`.

This routine is used for its side effects: it displays the generated figures and,
if `save_file = true`, also writes them to disk.

# Errors

* Throws an error if input lengths mismatch.
* Throws an error if no valid points remain after filtering.
* Throws an error if `fit_result.powers` is missing or its length does not match `fit_result.params`.
* Throws an error if the relative-error plot is requested with ``I_0 = 0``.
* Propagates errors from downstream plotting, file I/O, optional external PDF cropping,
  and internal linear-algebra operations used for covariance propagation.

# Examples

The example below shows the standard final plotting step after generating a raw
convergence dataset and a downstream fit result.

```julia
using Maranatha

f(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)

result = run_Maranatha(
    f,
    0.0, 1.0;
    dim = 4,
    nsamples = [2, 3, 4, 5, 6, 7, 8, 9],
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    err_method = :forwarddiff,
    fit_terms = 4,
    nerr_terms = 3,
    ff_shift = 0,
    use_threads = false,
    name_prefix = "4D_test",
    save_path = ".",
    write_summary = true
)

fit = least_chi_square_fit(
    result.a,
    result.b,
    result.h,
    result.avg,
    result.err,
    result.rule,
    result.boundary;
    nterms = result.fit_terms,
    ff_shift = result.ff_shift,
    nerr_terms = result.nerr_terms
)

print_fit_result(fit)

plot_convergence_result(
    result.a,
    result.b,
    "4D_test",
    result.h,
    result.avg,
    result.err,
    fit;
    rule = result.rule,
    boundary = result.boundary
)
```

If a long convergence study is executed in multiple partial runs
(for example, due to wall-time limits or interrupted sessions),
the saved `.jld2` files can later be merged with
[`Maranatha.Utils.MaranathaIO.merge_datapoint_result_files`](@ref)
or [`Maranatha.Utils.MaranathaIO.merge_datapoint_results`](@ref),
and the merged result can be passed to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref)
in exactly the same way as a single-run result.

```julia
using Maranatha

merged_path = merge_datapoint_result_files(
    "result_part1.jld2",
    "result_part2.jld2",
    "result_part3.jld2";
    output_path = "result_merged.jld2",
    write_summary = true,
)

merged = load_datapoint_results(merged_path)

fit = least_chi_square_fit(
    merged.a,
    merged.b,
    merged.h,
    merged.avg,
    merged.err,
    merged.rule,
    merged.boundary;
    nterms = merged.fit_terms,
    ff_shift = merged.ff_shift,
    nerr_terms = merged.nerr_terms
)
```

The merged output path may also be generated automatically from the
actual subdivision counts present in the merged result:

```julia
using Maranatha

merged_path = merge_datapoint_result_files(
    "result_part1.jld2",
    "result_part2.jld2",
    "result_part3.jld2";
    write_summary = true,
    output_dir = ".",
    name_prefix = "merged"
)

merged = load_datapoint_results(merged_path)
```

Then the output filename is automatically constructed as

```julia
result_merged_\$(rule)_\$(boundary)_N_2_3_4_5_6_7.jld2
```

Selected subdivision counts can also be removed from an existing result
file before fitting. This is useful when very coarse resolutions are
considered unreliable or visually inconsistent with the main trend.

```julia
using Maranatha

filtered_path = drop_nsamples_from_file(
    "result_full.jld2",
    [2, 3];
    write_summary = true,
    output_dir = ".",
    name_prefix = "filtered"
)
```

Then the returned `filtered_path` points to a new file containing the same result data but with the specified `N` values removed, so that only the remaining resolutions are included in the downstream fit.

```julia
[2,3,4,5,6,7] -> [4,5,6,7]
```

You may then load the filtered result and pass it to the fitting routine as usual:

```julia
filtered = load_datapoint_results(filtered_path)

fit = least_chi_square_fit(
    filtered.a,
    filtered.b,
    filtered.h,
    filtered.avg,
    filtered.err,
    filtered.rule,
    filtered.boundary;
    nterms = filtered.fit_terms,
    ff_shift = filtered.ff_shift,
    nerr_terms = filtered.nerr_terms
)
```

Before fitting, it can be useful to inspect only the raw datapoints in a
chosen `h^p` coordinate in order to check apparent linearity, oscillation,
or resolution-dependent irregularities.

```julia
using Maranatha

plot_datapoints_result(
    "merged_test",
    merged.h,
    merged.avg,
    merged.err;
    h_power = 4,
    xscale = :linear,
    yscale = :linear,
    ymode = :value,
    rule = merged.rule,
    boundary = merged.boundary,
)
```

A relative-difference diagnostic view can also be drawn on log-log axes
once a reference value is available:

```julia
using Maranatha

plot_datapoints_result(
    "merged_test",
    merged.h,
    merged.avg,
    merged.err;
    h_power = 4,
    xscale = :log,
    yscale = :log,
    ymode = :reldiff,
    reference_value = fit.estimate,
    rule = merged.rule,
    boundary = merged.boundary,
)
```

"""
function plot_convergence_result(
    a::Real,
    b::Real,
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector,
    fit_result;
    rule::Symbol = :gauss_p3,
    boundary::Symbol = :LU_ININ,
    figs_dir::String=".",
    save_file::Bool=false
)

    # ------------------------------------------------------------
    # Determine leading convergence power automatically
    # using composite NC residual model (midpoint expansion)
    # ------------------------------------------------------------

    # Use the smallest h (largest N) as representative for order detection
    # (assumes hs correspond to increasing resolution)
    Nref = round(Int, (b - a) / minimum(float.(hs)))

    # --- Input checks ---
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    # ------------------------------------------------------------
    # Determine x-axis power from fit (e.g. h^p)
    # ------------------------------------------------------------
    fit_powers = if hasproperty(fit_result, :powers)
        fit_result.powers
    else
        JobLoggerTools.error_benji("fit_result missing :powers (cannot infer convergence power)")
    end

    # first nonzero power
    lead_pow = fit_powers[2]   # index 1 is 0 (constant term)

    # x-axis = h^lead_pow
    hx = hs .^ lead_pow

    errors_val = [e.total for e in errors]

    errors_pos = abs.(errors_val)

    mask = (hx .> 0) .& isfinite.(hx) .& isfinite.(estimates) .& isfinite.(errors_pos)

    # h2p = h2[mask]
    hxp = hx[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    isempty(hxp) && JobLoggerTools.error_benji("No valid points to plot.")

    # --- New fit result structure ---
    pvec = fit_result.params
    I0      = fit_result.estimate
    I0_err  = fit_result.error_estimate

    # --- Build model automatically from params ---
    # Model: I(h) = I0 + C1*h^p + C2*h^(p+2) + ...
    Cov = fit_result.cov

    # [PATCH] enforce symmetry for numerical stability
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
        v = Vector{Float64}(undef, length(pvec))
        @inbounds for i in 1:length(pvec)
            pow = fit_powers[i]
            v[i] = (pow == 0) ? 1.0 : h^pow
        end
        return v
    end

    function model_and_err(h)
        φ = basis_vec(h)
        y = LinearAlgebra.dot(pvec, φ)

        # [PATCH] prediction variance = φ' Cov φ, clipped at 0
        var = LinearAlgebra.dot(φ, CovS * φ)
        # σ = sqrt(max(var, 0.0))
        σ = sqrt(abs(var))
        return y, σ
    end

    # --- Smooth curve including extrapolated point at x = 0 ---
    xmin = minimum(hxp)
    xmax = maximum(hxp)

    x_range_log = 10 .^ range(log10(xmin), log10(xmax); length=200)

    # prepend zero explicitly
    x_range = vcat(0.0, x_range_log)

    # model needs h, not x; x = h^lead_pow  =>  h = x^(1/lead_pow)
    h_range = x_range .^ (1.0 / Float64(lead_pow))

    y_fit = similar(h_range)
    y_err = similar(h_range)

    for i in eachindex(h_range)
        y_fit[i], y_err[i] = model_and_err(h_range[i])
    end

    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    # Fit curve
    ax.plot(x_range, y_fit; color="black", linewidth=2.5)

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
        # h2p, estp;
        hxp, estp;
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

    # ax.set_xlabel(raw"$h^2$")
    ax.set_xlabel("\$h^{$(lead_pow)}\$")
    ax.set_ylabel("\$I(h)\$")

    # ---- annotate plot 1 (I0 +/- err, chi2/dof) ----
    chisq = hasproperty(fit_result, :chisq) ? fit_result.chisq : NaN
    dof   = hasproperty(fit_result, :dof)   ? fit_result.dof   : NaN
    red   = (isfinite(chisq) && isfinite(dof) && dof != 0) ? chisq / dof : NaN

    # try pretty formatter
    txt_I0 = try
        I0_e2d = AvgErrFormatter.avgerr_e2d_from_float(I0, I0_err; latex_grouping=true)
        "\$I_0 = $I0_e2d\$"
    catch
        @sprintf("\$I_0 = %.7g \\pm %.7g\$", I0, I0_err)
    end

    txt1 = txt_I0 * "\n" * @sprintf("\$\\chi^2/\\mathrm{d.o.f.} = \\texttt{%.7g}\$", red)

    _smart_text_placement!(fig, ax;
        text=txt1,
        x_points=collect(hxp),
        y_points=collect(estp),
        yerr_points=collect(errp),
        x_curve=collect(x_range),
        y_curve=collect(y_fit),
        fontsize=11
    )

    display(fig)

    basename = "result_$(name)_$(String(rule))_$(String(boundary))_extrap"
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
    (absI0 > 0) || JobLoggerTools.error_benji("Relative-error plot requires nonzero I0 (got I0=$I0).")

    # --- Data for relative-error plot ---
    Δp   = estp .- I0
    relp = abs.(Δp) ./ absI0

    # 1σ error bar for relative error (independent σ_I and σ_I0)
    rel_errp = sqrt.( (errp ./ absI0).^2 .+ ((abs.(Δp) .* I0_err) ./ (absI0^2)).^2 )

    # log-log requires strictly positive y (and yerr positive/finite)
    mask2 = (relp .> 0) .& isfinite.(relp) .& isfinite.(rel_errp) .& (rel_errp .> 0) .&
            (hxp .> 0) .& isfinite.(hxp)

    hxp2 = hxp[mask2]
    rel2 = relp[mask2]
    rerr2 = rel_errp[mask2]

    isempty(hxp2) && JobLoggerTools.error_benji("No valid positive relative-error points for log-log plot.")

    # --- Fit curve + curve uncertainty band for relative error ---
    # Use same x_range_log as before (no zero for log-log)
    x_range2 = x_range_log
    h_range2 = x_range2 .^ (1.0 / Float64(lead_pow))

    rel_fit = similar(h_range2)
    rel_sig = similar(h_range2)

    for i in eachindex(h_range2)
        yh, σy = model_and_err(h_range2[i])
        Δ = yh - I0

        rel_fit[i] = abs(Δ) / absI0

        # 1σ for relative error curve (propagate σy and σ_I0)
        rel_sig[i] = sqrt( (σy / absI0)^2 + ((abs(Δ) * I0_err) / (absI0^2))^2 )
    end

    # --- Plot ---
    fig2, ax2 = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    ax2.plot(x_range2, rel_fit; color="red", linewidth=2.5)

    # --- Reference slope line (slope = 1 in h^p axis) ---
    # anchor point near middle of curve
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
        hxp2, rel2;
        yerr=rerr2,
        fmt="o",
        color="blue",
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="blue"
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_xlabel("\$h^{$(lead_pow)}\$")
    ax2.set_ylabel(raw"$|I(h)-I_0|/|I_0|$")

    # ---- annotate plot 2 (chi2/dof only) ----
    txt_I0 = try
        I0_e2d = AvgErrFormatter.avgerr_e2d_from_float(I0, I0_err; latex_grouping=true)
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

    basename = "result_$(name)_$(String(rule))_$(String(boundary))_reldiff"
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

"""
    plot_convergence_result(
        result,
        fit_result;
        name::String = "Maranatha",
        rule::Symbol = result.rule,
        boundary::Symbol = result.boundary,
        figs_dir::String = ".",
        save_file::Bool = false,
    ) -> Nothing

Convenience wrapper around [`plot_convergence_result`](@ref) that accepts a
Maranatha run result object.

# Function description

This method extracts the necessary fields from the `result` object returned by
[`Maranatha.Runner.run_Maranatha`](@ref) and forwards them to the primary

```julia
plot_convergence_result(a, b, name, hs, estimates, errors, fit_result; ...)
```

method.

This allows users to call the plotting routine directly from the run result
without manually unpacking its components.

# Arguments

`result`
: Result object returned by [`Maranatha.Runner.run_Maranatha`](@ref).

`fit_result`
: Fit result returned by [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).

# Keyword arguments

Same as the primary [`plot_convergence_result`](@ref) method.

# Returns

Nothing.

# Errors

Same as the primary [`plot_convergence_result`](@ref) method.
"""
function plot_convergence_result(
    result,
    fit_result;
    name::String = "Maranatha",
    rule::Symbol = result.rule,
    boundary::Symbol = result.boundary,
    figs_dir::String = ".",
    save_file::Bool = false,
)
    return plot_convergence_result(
        result.a,
        result.b,
        name,
        Vector{Float64}(result.h),
        Vector{Float64}(result.avg),
        result.err,
        fit_result;
        rule = rule,
        boundary = boundary,
        figs_dir = figs_dir,
        save_file = save_file,
    )
end