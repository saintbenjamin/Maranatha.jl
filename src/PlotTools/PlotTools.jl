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

using ..PyPlot
using ..LinearAlgebra
using ..Printf

using ..Utils.JobLoggerTools
using ..Utils.AvgErrFormatter
using ..Quadrature
using ..ErrorEstimate

export set_pyplot_latex_style, plot_convergence_result, plot_quadrature_coverage_1d

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
    _smart_text_placement!(
        fig,
        ax;
        text::AbstractString,
        x_points::Vector{Float64},
        y_points::Vector{Float64},
        x_curve::Vector{Float64}=Float64[],
        y_curve::Vector{Float64}=Float64[],
        yerr_points::Union{Nothing,Vector{Float64}}=nothing,
        fontsize::Real=10,
        prefer_order = (
            (0.98, 0.98, :top,    :right),
            (0.02, 0.98, :top,    :left),
            (0.98, 0.02, :bottom, :right),
            (0.02, 0.02, :bottom, :left),
            (0.50, 0.98, :top,    :center),
            (0.50, 0.02, :bottom, :center),
            (0.98, 0.50, :center, :right),
            (0.02, 0.50, :center, :left),
            (0.80, 0.98, :top,    :center),
            (0.20, 0.98, :top,    :center),
            (0.80, 0.02, :bottom, :center),
            (0.20, 0.02, :bottom, :center),
            (0.98, 0.75, :center, :right),
            (0.02, 0.75, :center, :left),
            (0.98, 0.25, :center, :right),
            (0.02, 0.25, :center, :left),
        )
    ) -> Nothing

Place a text box inside an axis while heuristically avoiding plotted data, fitted curves,
and vertical error bars.

This helper evaluates a list of candidate text-anchor locations given in axis-fraction
coordinates and chooses the location with the lowest overlap score. The score is computed
in display coordinates after rendering, so the decision is based on the actual pixel-space
bounding box of the text and the actual rendered positions of plotted objects.

The algorithm considers three kinds of possible collisions:

1. discrete data points `(x_points, y_points)`
2. a polyline curve `(x_curve, y_curve)`
3. vertical error-bar segments defined by `yerr_points`

Each candidate location is penalized when:

- the text box extends outside the axis bounding box
- a data point falls inside the text box
- an error bar intersects the text box
- a curve segment intersects the text box
- a plotted element comes very close to the text box even without intersecting it

A weak additional bias slightly prefers positions near the top or bottom edge over the
exact middle, but only when the geometric overlap scores are otherwise similar.

The selected text is finally drawn with a semi-transparent white rounded box for readability.

# Arguments
- `fig` : Matplotlib figure object used to access the renderer.
- `ax` : Matplotlib axis object on which the text is placed.
- `text::AbstractString` : Text to place.
- `x_points::Vector{Float64}` : X-coordinates of discrete data points.
- `y_points::Vector{Float64}` : Y-coordinates of discrete data points.
- `x_curve::Vector{Float64}=Float64[]` : X-coordinates of a curve or fitted line to avoid.
- `y_curve::Vector{Float64}=Float64[]` : Y-coordinates of a curve or fitted line to avoid.
- `yerr_points::Union{Nothing,Vector{Float64}}=nothing` : Optional vertical error magnitudes
  for each data point. If provided, vertical error-bar segments are included in the overlap test.
- `fontsize::Real=10` : Font size of the placed text.
- `prefer_order` : Ordered tuple of candidate anchor positions in axis-fraction coordinates,
  each given as `(xf, yf, va_sym, ha_sym)` where `xf` and `yf` are in `ax.transAxes`
  coordinates and `va_sym`, `ha_sym` are vertical/horizontal alignment symbols.

# Returns
- `Nothing` : The function mutates the plot by adding the chosen text annotation directly to `ax`.

# Notes
- All collision checks are performed in display space, not data space, so the heuristic adapts
  to the actual rendered aspect ratio and axis scaling.
- Candidate positions are tested by temporarily creating an invisible text object, measuring its
  rendered bounding box, and then removing it.
- The curve-avoidance logic assumes that `(x_curve, y_curve)` forms a polyline and checks each
  consecutive segment against the candidate text box.
- Error bars are currently treated as vertical line segments only.
- If no candidate survives meaningfully, the function falls back to the top-left corner
  `(0.02, 0.98, :top, :left)`.

# Examples
```julia
_smart_text_placement!(
    fig, ax;
    text = "fit: y = ax + b",
    x_points = xs,
    y_points = ys,
    x_curve = xfit,
    y_curve = yfit,
    yerr_points = yerr,
    fontsize = 11
)
```

"""
function _smart_text_placement!(
    fig, 
    ax;
    text::AbstractString,
    x_points::Vector{Float64},
    y_points::Vector{Float64},
    x_curve::Vector{Float64}=Float64[],
    y_curve::Vector{Float64}=Float64[],
    yerr_points::Union{Nothing,Vector{Float64}}=nothing,
    fontsize::Real=10,
    prefer_order = (
        (0.98, 0.98, :top,    :right),
        (0.02, 0.98, :top,    :left),
        (0.98, 0.02, :bottom, :right),
        (0.02, 0.02, :bottom, :left),
        (0.50, 0.98, :top,    :center),
        (0.50, 0.02, :bottom, :center),
        (0.98, 0.50, :center, :right),
        (0.02, 0.50, :center, :left),
        (0.80, 0.98, :top,    :center),
        (0.20, 0.98, :top,    :center),
        (0.80, 0.02, :bottom, :center),
        (0.20, 0.02, :bottom, :center),
        (0.98, 0.75, :center, :right),
        (0.02, 0.75, :center, :left),
        (0.98, 0.25, :center, :right),
        (0.02, 0.25, :center, :left),
    )
)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    function _inflate_bbox(bb, pad)
        return (
            bb.x0 - pad,
            bb.x1 + pad,
            bb.y0 - pad,
            bb.y1 + pad
        )
    end

    function _point_inside_bbox(px, py, x0, x1, y0, y1)
        return (x0 <= px <= x1) && (y0 <= py <= y1)
    end

    function _point_rect_distance(px, py, x0, x1, y0, y1)
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        return hypot(dx, dy)
    end

    function _segment_intersects_bbox(p1x, p1y, p2x, p2y, x0, x1, y0, y1)
        # quick accept: endpoint inside
        if _point_inside_bbox(p1x, p1y, x0, x1, y0, y1) ||
           _point_inside_bbox(p2x, p2y, x0, x1, y0, y1)
            return true
        end

        # Liang-Barsky style clipping
        dx = p2x - p1x
        dy = p2y - p1y

        p = (-dx, dx, -dy, dy)
        q = (p1x - x0, x1 - p1x, p1y - y0, y1 - p1y)

        u1 = 0.0
        u2 = 1.0

        for i in 1:4
            pi = p[i]
            qi = q[i]
            if pi == 0.0
                if qi < 0.0
                    return false
                end
            else
                t = qi / pi
                if pi < 0.0
                    u1 = max(u1, t)
                else
                    u2 = min(u2, t)
                end
                if u1 > u2
                    return false
                end
            end
        end

        return true
    end

    function _segment_rect_distance(p1x, p1y, p2x, p2y, x0, x1, y0, y1)
        if _segment_intersects_bbox(p1x, p1y, p2x, p2y, x0, x1, y0, y1)
            return 0.0
        end

        # approximate by endpoint distances + rect corner to segment distances
        function point_segment_distance(px, py, ax, ay, bx, by)
            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay
            vv = vx*vx + vy*vy
            if vv == 0.0
                return hypot(px - ax, py - ay)
            end
            t = (wx*vx + wy*vy) / vv
            t = clamp(t, 0.0, 1.0)
            qx = ax + t*vx
            qy = ay + t*vy
            return hypot(px - qx, py - qy)
        end

        dmin = min(
            _point_rect_distance(p1x, p1y, x0, x1, y0, y1),
            _point_rect_distance(p2x, p2y, x0, x1, y0, y1),
            point_segment_distance(x0, y0, p1x, p1y, p2x, p2y),
            point_segment_distance(x0, y1, p1x, p1y, p2x, p2y),
            point_segment_distance(x1, y0, p1x, p1y, p2x, p2y),
            point_segment_distance(x1, y1, p1x, p1y, p2x, p2y),
        )
        return dmin
    end

    # ------------------------------------------------------------
    # Transform points to display coordinates
    # ------------------------------------------------------------
    pts = isempty(x_points) ? zeros(0, 2) : ax.transData.transform(hcat(x_points, y_points))

    # errorbar vertical segments in display coords
    err_segments = Tuple{Float64,Float64,Float64,Float64}[]
    if yerr_points !== nothing && !isempty(x_points)
        ylo = y_points .- yerr_points
        yhi = y_points .+ yerr_points
        p_lo = ax.transData.transform(hcat(x_points, ylo))
        p_hi = ax.transData.transform(hcat(x_points, yhi))
        for i in eachindex(x_points)
            push!(err_segments, (p_lo[i,1], p_lo[i,2], p_hi[i,1], p_hi[i,2]))
        end
    end

    # curve polyline segments in display coords
    crv_segments = Tuple{Float64,Float64,Float64,Float64}[]
    if !isempty(x_curve) && length(x_curve) == length(y_curve) && length(x_curve) >= 2
        crv = ax.transData.transform(hcat(x_curve, y_curve))
        for i in 1:(size(crv,1)-1)
            push!(crv_segments, (crv[i,1], crv[i,2], crv[i+1,1], crv[i+1,2]))
        end
    end

    # axes bbox in display coords
    axbb = ax.get_window_extent(renderer=renderer)

    best = nothing
    best_score = Inf

    # pixel pads
    pad = 8.0
    near_pad = 18.0

    for (xf, yf, va_sym, ha_sym) in prefer_order
        # measure with final bbox included
        t = ax.text(
            xf, yf, text;
            transform=ax.transAxes,
            fontsize=fontsize,
            va=String(va_sym),
            ha=String(ha_sym),
            alpha=0.0,
            bbox=Dict(
                "boxstyle"  => "round,pad=0.35",
                "facecolor" => "white",
                "alpha"     => 0.8,
                "edgecolor" => "none"
            )
        )

        bb = t.get_window_extent(renderer=renderer)
        x0, x1, y0, y1 = _inflate_bbox(bb, pad)
        t.remove()

        score = 0.0

        # penalty if text box goes outside axes region
        if x0 < axbb.x0 || x1 > axbb.x1 || y0 < axbb.y0 || y1 > axbb.y1
            score += 1e6
        end

        # point overlap + near-miss penalty
        @inbounds for i in 1:size(pts, 1)
            px = pts[i,1]
            py = pts[i,2]

            if _point_inside_bbox(px, py, x0, x1, y0, y1)
                score += 500.0
            else
                d = _point_rect_distance(px, py, x0, x1, y0, y1)
                if d < near_pad
                    score += 20.0 * (1.0 - d / near_pad)
                end
            end
        end

        # errorbar overlap + near-miss penalty
        for (xA, yA, xB, yB) in err_segments
            d = _segment_rect_distance(xA, yA, xB, yB, x0, x1, y0, y1)
            if d == 0.0
                score += 1200.0
            elseif d < near_pad
                score += 60.0 * (1.0 - d / near_pad)
            end
        end

        # curve overlap + near-miss penalty
        for (xA, yA, xB, yB) in crv_segments
            d = _segment_rect_distance(xA, yA, xB, yB, x0, x1, y0, y1)
            if d == 0.0
                score += 120.0
            elseif d < near_pad
                score += 8.0 * (1.0 - d / near_pad)
            end
        end

        # slight bias toward higher positions, but very weak
        score += 0.1 * abs(0.5 - yf)

        if score < best_score
            best_score = score
            best = (xf, yf, va_sym, ha_sym)
        end
    end

    if best === nothing
        best = (0.02, 0.98, :top, :left)
    end

    xf, yf, va_sym, ha_sym = best

    ax.text(
        xf, yf, text;
        transform=ax.transAxes,
        fontsize=fontsize,
        va=String(va_sym),
        ha=String(ha_sym),
        bbox=Dict(
            "boxstyle"  => "round,pad=0.35",
            "facecolor" => "white",
            "alpha"     => 0.8,
            "edgecolor" => "none"
        )
    )

    return nothing
end

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

Plot convergence data ``I(h)`` against ``h^{p}``, overlay the reconstructed fit curve,
visualize the propagated fit-uncertainty band, and generate an additional log-log plot
of the relative extrapolation error.

# Function description
This routine is a visualization companion to
[`Maranatha.LeastChiSquareFit.least_chi_square_fit`](@ref).
It reconstructs the fitted convergence model directly from the stored fit result and
produces two figures:

1. a main convergence plot of ``I(h)`` versus ``h^{p}``
2. a log-log relative-error plot of ``\\dfrac{|I(h)-I_0|}{|I_0|}`` versus ``h^{p}``

Here ``p`` is taken as the first non-constant power stored in `fit_result.powers`,
namely `fit_result.powers[2]`.

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

This function does *not* refit the data. Instead, it reconstructs the model from:

* `fit_result.params`
* `fit_result.cov`
* `fit_result.estimate`
* `fit_result.error_estimate`
* `fit_result.powers`

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
  Error estimates for ``I(h)``. Absolute values are used for plotting.
* `fit_result` :
  Fit object expected to provide at least:

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

# Returns

* `nothing`

# Errors

* Throws an error if input lengths mismatch.
* Throws an error if no valid points remain after filtering.
* Throws an error if `fit_result.powers` is missing or its length does not match `fit_result.params`.
* Throws an error if the relative-error plot is requested with ``I_0 = 0``.
* Propagates errors from downstream plotting, file I/O, external cropping, and linear-algebra steps.
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
    plot_quadrature_coverage_1d(
        f,
        a::Real,
        b::Real,
        N::Int;
        rule::Symbol = :newton_p3,
        boundary::Symbol = :LU_ININ,
        ngrid_f::Int = 4000,
        ngrid_block::Int = 400,
        name::String = "demo",
        figs_dir::String = ".",
        save_file::Bool = false
    ) -> Nothing

Visualize **``1``-dimensional quadrature behavior** on ``[a,b]`` by plotting the true integrand ``f(x)``
and a **pedagogical representation** of how the selected rule contributes to the integral.

# What this routine draws

This function always draws:

1. A dense curve of the true integrand ``f(x)`` over ``[a,b]``.
2. Quadrature nodes/weights ``(xs, ws)`` obtained from
   [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref).
3. A text annotation displaying the quadrature sum
   ```math
   \\hat I = \\sum_j w_j f(x_j)
   ````

computed directly from the returned nodes/weights.

Rather than being fixed at a hard-coded corner, this annotation is placed using
[`_smart_text_placement!`](@ref), which heuristically searches candidate locations
and tries to avoid overlap with plotted curves and node samples.

Then it draws **one** of the following rule-specific visualizations:

## 1) B-spline rules (`is_bs == true`)

For B-spline quadrature rules, this plotter reconstructs the **actual spline curve**
implicitly assumed by the B-spline backend and fills its area:

* Sample data: `y_j = f(x_j)` at Greville nodes.
* Reconstruct spline coefficients by solving the same collocation system used by the rule.
* Plot the spline curve **piecewise per knot span** and fill the region under each span.
  (Each span gets its own color, and the fill color is matched to the span line color.)

This visualization is meant to show *what curve the quadrature backend is effectively integrating*.
The reconstructed spline curve is also passed to the smart annotation-placement helper
so that the quadrature-sum label avoids covering the visually relevant spline shape.

## 2) Non-B-spline rules (`is_bs == false`)

For Newton-Cotes and Gauss-family rules, this plotter uses a simple **mass-bar view**:

* Each quadrature contribution is ``w_i \\, f(x_i)``.
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

* `rule`, `boundary`:
  Quadrature rule selector forwarded to the backend.
* `ngrid_f`:
  Number of grid points used to draw the dense reference curve of the true integrand.
* `ngrid_block`:
  Number of grid points per knot span when drawing reconstructed B-spline pieces.
* `name`:
  Label used in the output filename stem.
* `figs_dir`:
  Directory used for saving the output PDF when `save_file=true`.
* `save_file`:
  If `true`, save the figure as a PDF file. If `pdfcrop` is available, the saved PDF
  is cropped automatically.

# Annotation placement

The quadrature-sum annotation is positioned automatically using
[`_smart_text_placement!`](@ref). For non-B-spline rules, the helper avoids overlap
with the true integrand curve and finite node samples. For B-spline rules, it instead
avoids overlap with the reconstructed spline curve and the sampled node values.

# Output

When `save_file=true`, saves:

```julia
pedagogical_1D_\$(name)_\$(String(rule))_\$(String(boundary))_N\$(N).pdf
```

under `figs_dir`.

If the external command `pdfcrop` is available, the saved PDF is cropped automatically.

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
    name::String = "demo",
    figs_dir::String=".",
    save_file::Bool=false
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

    # ------------------------------------------------------------
    # Smart annotation placement for quadrature sum
    # ------------------------------------------------------------
    txt_quad = "Quadrature sum = $(I_hat)"

    mask_nodes = isfinite.(xs) .& isfinite.(y_nodes)
    x_pts = Float64.(xs[mask_nodes])
    y_pts = Float64.(y_nodes[mask_nodes])

    if is_bs
        _smart_text_placement!(fig, ax;
            text=txt_quad,
            x_points=collect(x_pts),
            y_points=collect(y_pts),
            x_curve=collect(xg),
            y_curve=collect(ys_spl),
            fontsize=10
        )
    else
        _smart_text_placement!(fig, ax;
            text=txt_quad,
            x_points=collect(x_pts),
            y_points=collect(y_pts),
            x_curve=collect(xg),
            y_curve=collect(yg),
            fontsize=10
        )
    end

    display(fig)

    basename = "pedagogical_1D_$(name)_$(String(rule))_$(String(boundary))_N$(N)"
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

end  # module PlotTools