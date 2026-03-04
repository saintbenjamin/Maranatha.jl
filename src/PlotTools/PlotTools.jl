# ============================================================================
# src/PlotTools/PlotTools.jl
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
using ..Utils.JobLoggerTools
using ..Quadrature
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
        rule::Symbol = :newton_p3,
        boundary::Symbol = :LU_ININ
    ) -> Nothing

Plot convergence data ``I(h)`` against ``h^{p}`` (where the leading exponent `p`
is taken from `fit_result.powers`), overlay the fitted extrapolation curve,
and visualize a *fit uncertainty band* propagated from the parameter covariance.

# Function description
This routine is a visualization companion to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
It produces a convergence plot and saves it as a PNG file.

The ``x``-axis is ``h^{p}``, where ``p = \\texttt{fit_result.powers[2]}``
(the first non-constant exponent used by the fit), and the ``y``-axis is the raw
quadrature estimate ``I(h)`` with pointwise error bars (absolute values are used).

Although the ``x``-axis is plotted in ``h^{p}``, the fitted model is evaluated as a function of ``h``.
Internally, the routine builds a dense grid in the plotted ``x`` coordinate
(``x = h^{p}``), converts it back to ``h`` via ``h = x^{1/p}``,
and evaluates the fitted model and its propagated uncertainty on that `h` grid.

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

### Requirement: `fit_result.powers`

This routine requires `fit_result` to provide `powers` so that the basis used in
the fit can be reconstructed exactly (no refitting). If `fit_result.powers` is
missing, the function throws an error.

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
* the extrapolated point at ``h^{p} = 0`` with uncertainty `fit_result.error_estimate`.

A second figure is also produced: a log-log plot of the relative error ``\\dfrac{I(h) - I_0}{I_0}``
versus ``h^p``, including propagated error bars for points, a propagated uncertainty band
for the fitted curve, and a slope-1 reference guide line.

## Additional output: relative-error log-log plot

In addition to the main convergence plot, this routine also produces a second figure
showing the **relative convergence error** on log-log axes:
```math
r(h) = \\frac{|I(h)-I_0|}{|I_0|},
\\qquad x = h^{p}.
```

The fitted model is evaluated on the same dense grid (in ``x=h^{p}``) and converted back
to ``h`` via ``h=x^{1/p}`` to compute the curve:
```math
r_{\\text{fit}}(h) = \\frac{|I_{\\text{fit}}(h)-I_0|}{|I_0|}.
```

### Error bars for data points (propagated)

For each measured point, the plotted ``1 \\, \\sigma`` error bar on the relative error
is computed by **first-order uncertainty propagation**, assuming independent uncertainties
for the point estimate ``I(h)`` and the extrapolated limit ``I_0``:
```math
\\sigma_r^2 \\approx
\\left(\\frac{\\sigma_I}{|I_0|}\\right)^2
+
\\left(\\frac{|I(h)-I_0|}{|I_0|^2}\\,\\sigma_{I_0}\\right)^2,
```

where ``\\sigma_I`` is the provided pointwise error (from `errors`) and
``\\sigma_{I_0}`` is `fit_result.error_estimate`.

### Uncertainty band for the fitted curve (propagated)

The shaded band around the fitted relative-error curve corresponds to:
```math
r_{\\text{fit}}(h) \\pm \\sigma_{r,\\text{fit}}(h),
```
with
```math
\\sigma_{r,\\text{fit}}^2(h) \\approx
\\left(\\frac{\\sigma_{\\text{fit}}(h)}{|I_0|}\\right)^2
+
\\left(\\frac{|I_{\\text{fit}}(h)-I_0|}{|I_0|^2}\\,\\sigma_{I_0}\\right)^2,
```
where ``\\sigma_{\\text{fit}}(h)`` is obtained from the parameter covariance:
```math
\\sigma_{\\text{fit}}(h)^2 = \\varphi(h)^{\\mathsf{T}} V \\varphi(h) \\,.
```

### Reference slope guide

A gray dashed **reference line** with slope 1 in the ``x=h^{p}`` coordinate
is drawn as a visual guide for the expected leading-order behavior:

```math
r(h) \\propto h^{p} \\quad \\Longleftrightarrow \\quad r \\propto x.
```

## Output

Two output files are saved:
```julia
convergence_\$(name)_\$(rule)_\$(boundary).png
convergence_\$(name)_\$(rule)_\$(boundary)_rel_log.png
```

# Arguments

* `a`, `b`:
  Integration bounds (currently unused by this plotting routine; retained for API consistency).
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
  * `fit_result.powers` (required; used to reconstruct the fit basis exactly).

# Keyword arguments

* `rule`:
  Rule symbol used only for labeling the output filename.
* `boundary`:
  Boundary symbol used only for labeling the output filename.

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
    rule::Symbol = :newton_p3,
    boundary::Symbol = :LU_ININ
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

    errors_pos = abs.(errors)

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
    CovS = Symmetric(Matrix(Cov))

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
        y = dot(pvec, φ)

        # [PATCH] prediction variance = φ' Cov φ, clipped at 0
        var = dot(φ, CovS * φ)
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
    ax.set_ylabel("Integral Estimate")

    fig.tight_layout()

    outfile = "convergence_$(name)_$(String(rule))_$(String(boundary)).png"
    fig.savefig(outfile)
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

    fig2.tight_layout()

    outfile2 = "convergence_$(name)_$(String(rule))_$(String(boundary))_rel_log.png"
    fig2.savefig(outfile2)
    PyPlot.close(fig2)

    return nothing
end

# =============================================================================
# Add to: src/plot/PlotTools.jl   (inside module PlotTools)
# =============================================================================

using ..Quadrature   # <-- add this line in PlotTools module imports

export plot_quadrature_coverage_1d

"""
    plot_quadrature_coverage_1d(
        f,
        a::Real,
        b::Real,
        N::Int;
        rule::Symbol = :newton_p3,
        boundary::Symbol = :LU_ININ,
        ngrid_f::Int = 4000,
        ngrid_block::Int = 400,
        name::String = "coverage",
    ) -> Nothing

Visualize **``1``-dimensional quadrature behavior** on ``[a,b]`` by plotting the true integrand ``f(x)``
and an **pedagogical representation** of how the selected rule contributes to the integral.

# What this routine draws

This function always draws:

1. A dense curve of the true integrand ``f(x)`` over ``[a,b]``.
2. Quadrature nodes/weights ``(xs, ws)`` obtained from
   [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref).
3. A text annotation:
   ```math
   \\hat I = \\sum_j w_j f(x_j)
   ```

computed directly from the returned nodes/weights.

Then it draws **one** of the following rule-specific visualizations:

## 1) B-spline rules (`is_bs == true`)

For B-spline quadrature rules, this plotter reconstructs the **actual spline curve**
implicitly assumed by your B-spline backend and fills its area:

* Sample data: ``y_j = f(x_j)`` at Greville nodes.
* Reconstruct spline coefficients by solving the same collocation system used by the rule.
* Plot the spline curve **piecewise per knot span** and fill the region under each span.
  (Each span gets its own color, and the fill color is matched to the span line color.)

This visualization is meant to show *what curve the quadrature backend is effectively integrating*.

## 2) Non-B-spline rules (`is_bs == false`)

For Newton-Cotes and Gauss-family rules, this plotter uses a simple **mass-bar view**:

* Each quadrature contribution is ``w_i f(x_i)``.
* A rectangle is drawn whose:

  * **width** is ``w_i``,
  * **height** is ``f(x_i)``,
    so that its signed area equals ``w_i \\, f(x_i)``.

Implementation detail:

* Bars are placed **sequentially from ``a``**, so the *bar ``x``-positions are not the original node locations*.
  This is an intentionally minimal-assumption visualization that uses only the available `(xs, ws)`.

Negative weights:

* If ``w_i < 0``, the interval would geometrically "go backwards".
  This implementation flips the drawn interval so it is always left-to-right, and flips the height sign,
  preserving the signed area ``w_i \\, f(x_i)``.

# Rule-family detection

The rule family is detected by:

* [`Maranatha.Quadrature.NewtonCotes._is_newton_cotes_rule`](@ref)
* [`Maranatha.Quadrature.Gauss._is_gauss_rule`](@ref)
* [`Maranatha.Quadrature.BSpline._is_bspline_rule`](@ref)

The `(rule, boundary)` pair is passed unchanged to the quadrature dispatcher, and any
validity constraints (e.g., composability constraints for composite rules) are enforced
by the backend.

# Arguments

* `f`: scalar integrand. If ``f(x)`` is not finite at a node, that node is skipped in ``\\hat I``.
* `a`, `b`: interval endpoints (finite, with ``b > a``).
* `N`: subdivision count forwarded to the backend.

# Keyword arguments

* `rule`, `boundary`: quadrature rule selector forwarded to the backend.
* `ngrid_f`: number of points for drawing the true integrand curve.
* `ngrid_block`: number of points per knot span (B-spline piecewise drawing only).
* `name`: label used in the output PNG filename.

# Output

Saves:

```julia
quad_coverage_\$(name)_\$(String(rule))_\$(String(boundary))_N\$(N).png
```

# Returns

`nothing`.
"""
function plot_quadrature_coverage_1d(
    f,
    a::Real,
    b::Real,
    N::Int;
    rule::Symbol = :newton_p3,
    boundary::Symbol = :LU_ININ,
    ngrid_f::Int = 4000,
    ngrid_block::Int = 400,
    name::String = "coverage",
)::Nothing

    # -------------------------------
    # Basic checks
    # -------------------------------
    (N isa Int) || JobLoggerTools.error_benji("N must be Int (got $(typeof(N)))")
    N >= 1 || JobLoggerTools.error_benji("N must be ≥ 1 (got N=$N)")

    aa = Float64(a)
    bb = Float64(b)
    (isfinite(aa) && isfinite(bb)) || JobLoggerTools.error_benji("a,b must be finite (got a=$a, b=$b)")
    (bb > aa) || JobLoggerTools.error_benji("Require b > a (got a=$a, b=$b)")

    # Identify rule family (must match your existing modules)
    is_ns   = Quadrature.NewtonCotes._is_newton_cotes_rule(rule)
    is_gaus = Quadrature.Gauss._is_gauss_rule(rule)
    is_bs   = Quadrature.BSpline._is_bspline_rule(rule)
    (is_ns || is_gaus || is_bs) || JobLoggerTools.error_benji("Unsupported rule family: rule=$rule")

    # Build global nodes/weights (single source of truth)
    xs, ws = Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)
    (length(xs) == length(ws)) || JobLoggerTools.error_benji("Internal: xs/ws length mismatch")

    # -------------------------------
    # Helpers: safe f eval
    # -------------------------------
    @inline function _f_float(x::Float64)::Float64
        y = f(x)
        return (y isa Real && isfinite(y)) ? Float64(y) : NaN
    end

    # -------------------------------
    # BSpline: reconstruct spline s(x) from samples y_j = f(xs[j])
    # so we can plot the actual assumed curve and fill its area.
    # -------------------------------
    function _bspline_reconstruct_coeffs(
        xs::Vector{Float64},
        y::Vector{Float64},
        a::Float64,
        b::Float64,
        N::Int,
        rule::Symbol,
        boundary::Symbol
    )::Tuple{Vector{Float64}, Int, Vector{Float64}}
        p    = Quadrature.BSpline._parse_bspline_p(rule)
        kind = Quadrature.BSpline._bspline_kind(rule)  # :interp or :smooth

        t  = Quadrature.BSpline._build_knots_uniform(a, b, N, p, boundary)
        nb = length(t) - p - 1
        (length(xs) == nb) || JobLoggerTools.error_benji("BSpline internal mismatch: length(xs)=$(length(xs)) nb=$nb")

        # Build A[j,i] = B_i(xs[j])
        A = Matrix{Float64}(undef, nb, nb)
        @inbounds for j in 1:nb
            Bj = Quadrature.BSpline._bspline_basis_all(xs[j], t, p)
            @inbounds for i in 1:nb
                A[j,i] = Bj[i]
            end
        end

        if kind === :interp
            c = A \ y
            return t, p, c
        else
            # Keep consistent with your dispatch (λ fixed to 0.0 for now)
            λ = 0.0
            R = Quadrature.BSpline._roughness_R_second_diff(nb)
            M = transpose(A) * A + λ * R
            c = M \ (transpose(A) * y)
            return t, p, c
        end
    end

    @inline function _bspline_eval(
        x::Float64,
        t::Vector{Float64},
        p::Int,
        c::Vector{Float64}
    )::Float64
        B = Quadrature.BSpline._bspline_basis_all(x, t, p)
        return dot(c, B)
    end

    # -------------------------------
    # Plot
    # -------------------------------
    set_pyplot_latex_style(0.55)
    fig, ax = PyPlot.subplots(figsize=(6.2, 4.8), dpi=450)

    # Dense integrand curve (true f)
    xg = collect(range(aa, bb; length=ngrid_f))
    yg = Vector{Float64}(undef, length(xg))
    @inbounds for i in eachindex(xg)
        yg[i] = _f_float(xg[i])
    end
    ax.plot(xg, yg; linewidth=2.2)

    # Node samples (for scatter + for quadrature sum)
    y_nodes = Vector{Float64}(undef, length(xs))
    @inbounds for j in eachindex(xs)
        y_nodes[j] = _f_float(Float64(xs[j]))
    end

    # Scatter only finite nodes
    # mask_nodes = isfinite.(y_nodes)
    # xs_plot = xs[mask_nodes]
    # ys_plot = y_nodes[mask_nodes]

    # # Marker size ~ |w|
    # wabs = abs.(ws)
    # wmax = maximum(wabs)
    # ms_all = (wmax > 0) ? (6.0 .+ 18.0 .* (wabs ./ wmax)) : fill(6.0, length(ws))
    # ms_plot = ms_all[mask_nodes]

    # ax.scatter(xs_plot, ys_plot; s=ms_plot, alpha=0.9)

    # Quadrature sum text (same meaning as before, but no contrib vector)
    I_hat = 0.0
    @inbounds for j in eachindex(xs)
        yj = y_nodes[j]
        wj = ws[j]
        if isfinite(yj) && isfinite(wj)
            I_hat += Float64(wj) * Float64(yj)
        end
    end
    ax.text(
        0.02, 0.98,
        "Quadrature sum = $(I_hat)",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left"
    )

    # ------------------------------------------------------------
    # BSpline: plot the actual spline curve s(x) and fill under it,
    # panel-by-panel using knot spans intersecting [a,b].
    # ------------------------------------------------------------
    if is_bs
        all(isfinite.(y_nodes)) || JobLoggerTools.error_benji(
            "BSpline coverage plot requires all node samples finite (got NaN/Inf)."
        )

        t, p, c = _bspline_reconstruct_coeffs(Float64.(xs), Float64.(y_nodes), aa, bb, N, rule, boundary)

        # Spline curve on dense grid
        ys_spl = Vector{Float64}(undef, length(xg))
        @inbounds for i in eachindex(xg)
            ys_spl[i] = _bspline_eval(xg[i], t, p, c)
        end
        # ax.plot(xg, ys_spl; linewidth=1.6, alpha=0.9)

        # Fill per knot span (only spans with positive length inside [a,b])
        nt = length(t)
        @inbounds for k in 1:(nt-1)
            x0 = t[k]
            x1 = t[k+1]
            (x1 > x0) || continue

            L = max(x0, aa)
            R = min(x1, bb)
            (R > L) || continue

            xb = collect(range(L, R; length=ngrid_block))
            yb = Vector{Float64}(undef, length(xb))
            @inbounds for i in eachindex(xb)
                yb[i] = _bspline_eval(xb[i], t, p, c)
            end
            line = ax.plot(xb, yb; linewidth=1.6, alpha=0.9)[1]
            col  = line."get_color"()
            ax.fill_between(xb, zeros(length(xb)), yb; alpha=0.22, linewidth=0.0, color=col)
        end
    else
        # ------------------------------------------------------------
        # Width = weight interpretation (Δx_i = w_i), height = f(x_i)
        # - Educational "mass bar" view.
        # - Applies to non-BSpline rules here (same as your current behavior).
        # ------------------------------------------------------------

        # Keep only finite (x, w, y)
        xs_c = Float64[]
        ws_c = Float64[]
        ys_c = Float64[]
        @inbounds for j in eachindex(xs)
            xj = Float64(xs[j])
            wj = Float64(ws[j])
            yj = Float64(y_nodes[j])
            if isfinite(xj) && isfinite(wj) && isfinite(yj)
                push!(xs_c, xj)
                push!(ws_c, wj)
                push!(ys_c, yj)
            end
        end

        isempty(xs_c) || begin
            # Sort by x to keep visual order
            perm = sortperm(xs_c)
            xs_c = xs_c[perm]
            ws_c = ws_c[perm]
            ys_c = ys_c[perm]

            # Sequential placement starting at a
            x_left = aa

            @inbounds for i in 1:length(xs_c)
                width = ws_c[i]
                width == 0.0 && continue

                xL = x_left
                xR = x_left + width

                height = ys_c[i]

                # If negative width, flip interval and flip sign so area stays w*f
                if xR < xL
                    xL, xR = xR, xL
                    height = -height
                end

                ax.fill_between(
                    [xL, xR],
                    [0.0, 0.0],
                    [height, height];
                    alpha=0.35,
                    linewidth=0.8,
                    edgecolor="black"
                )

                x_left = x_left + width
            end

            ax.axhline(0.0; linewidth=0.9, alpha=0.6)
        end
    end

    ax.set_xlim(aa, bb)
    ax.set_xlabel(raw"$x$")
    ax.set_ylabel(is_ns ? raw"$f(x)$ / block interpolants" : raw"$f(x)$ and discrete contributions")
    ax.set_title("$(String(rule)), $(String(boundary)), N=$N")

    fig.tight_layout()
    outfile = "quad_coverage_$(name)_$(String(rule))_$(String(boundary))_N$(N).png"
    fig.savefig(outfile)
    PyPlot.close(fig)

    return nothing
end

end  # module PlotTools