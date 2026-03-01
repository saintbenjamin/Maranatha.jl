# ============================================================================
# src/figs/PlotTools.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PlotTools

using PyPlot
using LinearAlgebra
using ..JobLoggerTools
using ..ErrorEstimate

export plot_convergence_result, set_pyplot_latex_style

"""
    set_pyplot_latex_style(
        scale::Float64=0.5
    ) -> Nothing

Configure [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering and appropriate font settings for publications.

This function modifies [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) to enable [``\\LaTeX``](https://www.latex-project.org/)-based text rendering and adjust 
font sizes, marker sizes, and line widths for consistent visual output.  
Useful for generating high-quality plots for papers or presentations.

# Arguments
- `scale::Float64`: Scaling factor for font sizes and figure dimensions. Default is `0.5`.

# Side Effects
- Modifies [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)'s global rendering configuration via [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams).
"""
function set_pyplot_latex_style(
    scale::Float64=0.5
)

    mpl = PyPlot.matplotlib

    rcParams = mpl."rcParams"
    rcParams.update(mpl."rcParamsDefault")

    rcParams.update(Dict(
        "figure.figsize" => (16 * scale, 12 * scale),
        "font.size" => 24 * scale,
        "axes.labelsize" => 24 * scale,
        "legend.fontsize" => 24 * scale,
        "lines.markersize" => 18 * scale,
        "lines.linewidth" => 4 * scale,
        "font.family" => "lmodern",
        "text.usetex" => true,
        "text.latex.preamble" => raw"\usepackage{lmodern}",
        "axes.grid" => true,
        "grid.alpha" => 0.3
    ))
    return nothing
end

"""
    plot_convergence_result(
        a::Real,
        b::Real,
        name::String,
        hs::Vector{Float64},
        estimates::Vector{Float64},
        errors::Vector{Float64},
        fit_result;
        rule::Symbol = :ns_p3,
        boundary::Symbol = :LCRC
    ) -> Nothing

Plot convergence data ``I(h)`` against ``h^2``, overlay the fitted extrapolation curve,
and visualize a *fit uncertainty band* propagated from the parameter covariance.

# Function description
This routine is a visualization companion to [`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
It produces a convergence plot and saves it as a PNG file.

The ``x``-axis is ``h^2`` (with ``h`` provided via `hs`), and the ``y``-axis is the raw quadrature
estimate ``I(h)`` with pointwise error bars (absolute values are used for plotting).

Although the ``x``-axis is plotted in ``h^2``, the fitted model is evaluated as a function of ``h``.
Internally, the routine builds a dense grid in ``h^2``, converts it via ``h = \\sqrt{h^2}``,
and evaluates the model and its propagated uncertainty on that ``h`` grid.

## Model reconstruction (no refitting)
This function does *not* refit anything. It reconstructs the model from the stored fit output:
- `pvec   = fit_result.params`
- `Cov    = fit_result.cov`
- `I0     = fit_result.estimate`
- `I0_err = fit_result.error_estimate`

A convergence model is reconstructed using the exponent vector stored in `fit_result.powers`:
```math
I(h) = \\bm{\\lambda}^{\\mathsf{T}} \\varphi(h),
\\qquad
\\varphi_1(h)=1,
\\qquad
\\varphi_i(h)=h^{\\texttt{powers[i]}} \\ (i\\ge 2),
```
where `powers = fit_result.powers` and `length(powers) == length(pvec)` is required.

### Fallback behavior

If `fit_result.powers` is not present, the routine falls back to a legacy assumption:
it infers a representative subdivision count `Nref` from the smallest `h` in `hs`, extracts a leading
residual order `k` via [`Maranatha.ErrorEstimate._leading_midpoint_residual_term`](@ref),
sets `p = k`, and then uses:
```math
\\texttt{powers = (0, p, p+2, p+4, \\ldots)}
```
to match the length of `fit_result.params`.

In this fallback mode, `rule` and `boundary` are used only for that residual-based power inference
and for labeling the output filename.

## Fit curve and uncertainty band

For each point on a dense grid in `h`, the basis vector is constructed using the same exponents
used by the fit (preferably `fit_result.powers`).

The routine evaluates:

* fit curve: ``I_{\\text{fit}}(h) = \\bm{\\lambda}^{\\mathsf{T}} \\varphi(h)``
* ``1 \\, \\sigma`` prediction uncertainty from parameter covariance:
```math
\\sigma_{\\text{fit}}(h)^2 = \\varphi(h)^{\\mathsf{T}} \\, V \\, \\varphi(h),
```
where `V` is `fit_result.cov`. The plotted shaded band corresponds to
``I_{\\text{fit}}(h) \\pm \\sigma_{\\text{fit}}(h)`` and includes parameter correlations.

## Plot elements

The resulting figure contains:

* the fitted curve ``I_{\\text{fit}}(h)`` (line),
* the fit uncertainty band ``\\pm \\sigma`` (shaded region),
* the measured points with error bars,
* the extrapolated point at ``h^2 = 0`` with uncertainty `fit_result.error_estimate`.

## Output

The output file is saved as:

```julia
convergence_\$(name)_\$(rule)_\$(boundary).png
```

# Arguments

* `a`, `b`:
  Integration bounds used only in the legacy fallback path to derive `Nref` from the smallest step size in `hs`.
* `name`:
  Label used in the output filename.
* `hs`:
  Step sizes `h` (typically ``\\displaystyle{h=\\frac{b-a}{N}}``).
* `estimates`:
  Quadrature estimates ``I(h)`` corresponding to `hs`.
* `errors`:
  Error estimates for ``I(h)`` (absolute values are used for plotting).
* `fit_result`:
  Fit object expected to provide:

  * `fit_result.params`
  * `fit_result.cov`
  * `fit_result.estimate`
  * `fit_result.error_estimate`
    Optionally:
  * `fit_result.powers` (recommended; required for exact basis reconstruction).

# Keyword arguments

* `rule`:
  Composite Newton-Cotes rule symbol (must be `:ns_pK` style).
* `boundary`:
  Boundary pattern (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).

# Returns

* `nothing`.

# Errors

* Throws an error if input lengths mismatch.
* Throws an error if no valid points remain after filtering.
* Throws an error if `fit_result.powers` exists but has a length mismatch with `fit_result.params`.
* Propagates errors from downstream plotting and linear-algebra operations
  (e.g. non-finite values after filtering, covariance not usable for propagation).
"""
function plot_convergence_result(
    a::Real,
    b::Real,
    name::String,
    hs::Vector{Float64},
    estimates::Vector{Float64},
    errors::Vector{Float64},
    fit_result;
    rule::Symbol = :ns_p3,
    boundary::Symbol = :LCRC
)

    # ------------------------------------------------------------
    # Determine leading convergence power automatically
    # using composite NC residual model (midpoint expansion)
    # ------------------------------------------------------------

    # Use the smallest h (largest N) as representative for order detection
    # (assumes hs correspond to increasing resolution)
    Nref = round(Int, (b - a) / minimum(float.(hs)))

    # k, _ = ErrorEstimate._leading_midpoint_residual_term(rule, boundary, Nref)
    # p = k

    # # --- Determine leading power from rule ---
    # p =
    #     rule == :simpson13_close ? 4 :
    #     rule == :simpson13_open  ? 4 :
    #     rule == :simpson38_close ? 4 :
    #     rule == :simpson38_open  ? 4 :
    #     rule == :bode_close      ? 6 :
    #     rule == :bode_open       ? 6 :
    #     JobLoggerTools.error_benji("Unknown rule")

    # --- Input checks ---
    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    # Raw x = h^2
    h2 = hs .^ 2
    errors_pos = abs.(errors)

    mask = (h2 .> 0) .& isfinite.(h2) .& isfinite.(estimates) .& isfinite.(errors_pos)

    h2p = h2[mask]
    estp = estimates[mask]
    errp = errors_pos[mask]

    isempty(h2p) && JobLoggerTools.error_benji("No valid points to plot.")

    # --- New fit result structure ---
    pvec = fit_result.params
    I0      = fit_result.estimate
    I0_err  = fit_result.error_estimate

    # --- Build model automatically from params ---
    # Model: I(h) = I0 + C1*h^p + C2*h^(p+2) + ...
    Cov = fit_result.cov

    # [PATCH] enforce symmetry for numerical stability
    CovS = Symmetric(Matrix(Cov))

    # --------------------------------------------
    # Determine model exponents used by the fit
    # Prefer fit_result.powers if present; otherwise fall back.
    # --------------------------------------------
    fit_powers = if hasproperty(fit_result, :powers)
        fit_result.powers
    else
        # fallback: legacy assumption (p, p+2, p+4, ...)
        Nref = round(Int, (b - a) / minimum(float.(hs)))
        k, _ = ErrorEstimate._leading_midpoint_residual_term(rule, boundary, Nref)
        p = k
        vcat(0, [p + 2*(i-2) for i in 2:length(pvec)])
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
        y = dot(pvec, φ)

        # [PATCH] prediction variance = φ' Cov φ, clipped at 0
        var = dot(φ, CovS * φ)
        # σ = sqrt(max(var, 0.0))
        σ = sqrt(abs(var))
        return y, σ
    end

    # --- Smooth curve including extrapolated point at h^2 = 0 ---
    h2min = minimum(h2p)
    h2max = maximum(h2p)

    h2_range_log = 10 .^ range(log10(h2min), log10(h2max); length=200)

    # prepend zero explicitly
    h2_range = vcat(0.0, h2_range_log)

    # model needs h, not h^2
    h_range  = sqrt.(h2_range)
    y_fit = similar(h_range)
    y_err = similar(h_range)

    for i in eachindex(h_range)
        y_fit[i], y_err[i] = model_and_err(h_range[i])
    end


    # Style
    set_pyplot_latex_style(0.5)

    fig, ax = PyPlot.subplots(figsize=(5.6,5.0), dpi=500)

    # Fit curve
    ax.plot(h2_range, y_fit; color="black", linewidth=2.5)

    # --- Fit error band ---
    ax.fill_between(
        h2_range,
        y_fit .- y_err,
        y_fit .+ y_err;
        alpha=0.25,
        linewidth=0,
        color="black"
    )

    # Data points
    ax.errorbar(
        h2p, estp;
        yerr=errp,
        fmt="o",
        color="blue",
        capsize=6,
        markerfacecolor="none",
        markeredgecolor="blue"
    )

    # --- Extrapolated point at h^2 = 0 ---
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

    ax.set_xlabel(raw"$h^2$")
    ax.set_ylabel("Integral Estimate")

    fig.tight_layout()

    outfile = "convergence_$(name)_$(String(rule))_$(String(boundary)).png"
    fig.savefig(outfile)
    PyPlot.close(fig)

    return nothing
end

end  # module PlotTools