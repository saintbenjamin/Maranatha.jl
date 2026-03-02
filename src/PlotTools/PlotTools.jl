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
residual order `k` via [`Maranatha.ErrorEstimate.ErrorNewtonCotes._leading_midpoint_residual_term`](@ref),
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
        k, _ = ErrorEstimate.NewtonCotes._leading_midpoint_residual_term(rule, boundary, Nref)
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
        rule::Symbol = :ns_p3,
        boundary::Symbol = :LCRC,
        ngrid_f::Int = 4000,
        ngrid_block::Int = 400,
        name::String = "coverage",
        show_bars::Bool = true,
        bar_width_scale::Float64 = 1.0,
        show_block_fill::Bool = false
    ) -> Nothing

Educational 1D visualization:

1) Plot the integrand `f(x)` on `[a,b]`.
2) Reconstruct the composite `:ns_pK` block tiling implied by `(N, boundary)`.
3) For each block, build the degree-(p-1) polynomial interpolant through the block nodes,
   and visualize the "covered region" by filling the area under that interpolant.

This makes it easy to see how increasing `N` (resolution) and `K` (local node count)
changes the piecewise polynomial approximation quality.

The plot is saved as:
`quad_coverage_\$(name)_\$(rule)_\$(boundary)_N\$(N).png`.
"""
function plot_quadrature_coverage_1d(
    f,
    a::Real,
    b::Real,
    N::Int;
    rule::Symbol = :ns_p3,
    boundary::Symbol = :LCRC,
    ngrid_f::Int = 4000,
    ngrid_block::Int = 400,
    name::String = "coverage",
    show_bars::Bool = true,
    bar_width_scale::Float64 = 1.0,
    show_block_fill::Bool = false
)::Nothing

    # -------------------------------
    # Basic checks
    # -------------------------------
    N >= 1 || JobLoggerTools.error_benji("N must be ≥ 1 (got N=$N)")
    startswith(String(rule), "ns_p") || JobLoggerTools.error_benji("This plotter supports only :ns_pK rules (got rule=$rule)")
    p = parse(Int, String(rule)[5:end])
    p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2 (got p=$p from rule=$rule)")

    # -------------------------------
    # Decode boundary and tiling
    # -------------------------------
    @inline function _decode_boundary(boundary::Symbol)
        if boundary === :LCRC
            return (:closed, :closed)
        elseif boundary === :LORC
            return (:opened, :closed)
        elseif boundary === :LCRO
            return (:closed, :opened)
        elseif boundary === :LORO
            return (:opened, :opened)
        else
            JobLoggerTools.error_benji("boundary must be one of :LCRC | :LORC | :LCRO | :LORO (got $boundary)")
        end
    end

    @inline function _local_width(p::Int, kind::Symbol)::Int
        kind === :closed && return p - 1
        kind === :opened && return p
        JobLoggerTools.error_benji("unknown local kind=$kind")
    end

    @inline function _check_condition(p::Int, boundary::Symbol, Nsub::Int)
        Ltype, Rtype = _decode_boundary(boundary)
        wL = _local_width(p, Ltype)
        wR = _local_width(p, Rtype)
        wC = p - 1

        if Nsub < wL + wR
            JobLoggerTools.error_benji(
                "Invalid N for boundary=$boundary: need N ≥ wL+wR = $(wL+wR), got N=$Nsub"
            )
        end

        rem = Nsub - wL - wR
        if rem % wC != 0
            m_low  = rem ÷ wC
            m_high = m_low + 1
            N_low  = wL + m_low  * wC + wR
            N_high = wL + m_high * wC + wR
            JobLoggerTools.error_benji(
                "Invalid N for boundary=$boundary.\n" *
                "Require: N = wL + m*(p-1) + wR.\n" *
                "Got: N=$Nsub, wL=$wL, wR=$wR, (p-1)=$wC.\n" *
                "Nearest valid N: $N_low or $N_high."
            )
        end

        m = rem ÷ wC
        return m, wL, wR, Ltype, Rtype
    end

    m, wL, wR, Ltype, Rtype = _check_condition(p, boundary, N)

    # -------------------------------
    # Build global nodes/weights (for markers)
    # -------------------------------
    xs, ws = Quadrature.quadrature_1d_nodes_weights(a, b, N, rule, boundary)

    aa = Float64(a)
    bb = Float64(b)
    h = (bb - aa) / Float64(N)

    # -------------------------------
    # Helper: polynomial interpolant through (xj, yj)
    # (simple Vandermonde; educational, not numerically optimal)
    # -------------------------------
    function _poly_fit_coeffs(xv::Vector{Float64}, yv::Vector{Float64})
        n = length(xv)
        V = Matrix{Float64}(undef, n, n)
        @inbounds for i in 1:n
            V[i, 1] = 1.0
            for k in 2:n
                V[i, k] = V[i, k-1] * xv[i]
            end
        end
        return V \ yv
    end

    function _poly_eval(c::Vector{Float64}, x::Float64)
        # Horner
        y = 0.0
        @inbounds for k in length(c):-1:1
            y = y * x + c[k]
        end
        return y
    end

    # -----------------------------------------
    # Lagrange basis values L[i,k] = ℓ_i(xb[k])
    # Use barycentric form (stable + fast enough)
    # -----------------------------------------
    function _lagrange_basis_matrix(xv::Vector{Float64}, xb::Vector{Float64})
        p = length(xv)
        nb = length(xb)

        # barycentric weights
        w = ones(Float64, p)
        @inbounds for j in 1:p
            prod = 1.0
            xj = xv[j]
            for m in 1:p
                m == j && continue
                prod *= (xj - xv[m])
            end
            w[j] = 1.0 / prod
        end

        L = Matrix{Float64}(undef, p, nb)

        @inbounds for k in 1:nb
            x = xb[k]

            # if x coincides with a node, basis is exact unit vector
            hit = 0
            for j in 1:p
                if x == xv[j]
                    hit = j
                    break
                end
            end
            if hit != 0
                for j in 1:p
                    L[j,k] = (j == hit) ? 1.0 : 0.0
                end
                continue
            end

            denom = 0.0
            tmp = Vector{Float64}(undef, p)
            for j in 1:p
                t = w[j] / (x - xv[j])
                tmp[j] = t
                denom += t
            end
            for j in 1:p
                L[j,k] = tmp[j] / denom
            end
        end

        return L
    end

    # -------------------------------
    # Plot integrand and coverage
    # -------------------------------
    set_pyplot_latex_style(0.55)
    fig, ax = PyPlot.subplots(figsize=(6.2, 4.8), dpi=450)

    # Get matplotlib default color cycle (for block coloring)
    # Robust color cycle getter (does not depend on rcParams["axes.prop_cycle"])
    function _get_cycle_colors(ax; n::Int=10)
        cols = String[]
        for _ in 1:n
            c = ax."_"*"get_lines"()."get_next_color"()  # <- NO, this is wrong in Julia
        end
        return cols
    end

    # Dense integrand curve
    xg = collect(range(aa, bb; length=ngrid_f))
    yg = Vector{Float64}(undef, length(xg))
    @inbounds for i in eachindex(xg)
        yi = f(xg[i])
        yg[i] = (isfinite(yi) ? Float64(yi) : NaN)
    end
    ax.plot(xg, yg; linewidth=2.2)

    # Scatter nodes (marker size ~ |w|, purely for intuition)
    wabs = abs.(ws)
    wmax = maximum(wabs)
    ms = if wmax > 0
        6.0 .+ 18.0 .* (wabs ./ wmax)
    else
        fill(6.0, length(ws))
    end
    ax.scatter(xs, [f(xi) for xi in xs]; s=ms, alpha=0.9)

    # ------------------------------------------------------------
    # Quadrature bar data (draw later with per-block colors)
    # area(bar_j) = height_j * (h*bar_width_scale) = ws_j * f(xs_j)
    # ------------------------------------------------------------
    bw = h * bar_width_scale

    heights = fill(NaN, length(xs))   # indexed by node j=0..N stored at j+1
    contrib = fill(NaN, length(xs))

    I_hat = NaN
    if show_bars
        @inbounds for j in eachindex(xs)
            yi = f(xs[j])
            if isfinite(yi) && isfinite(ws[j]) && isfinite(bw) && bw != 0.0
                contrib[j] = ws[j] * Float64(yi)
                heights[j] = contrib[j] / bw
            end
        end
        mask_bar = isfinite.(contrib)
        I_hat = sum(contrib[mask_bar])

        ax.text(
            0.02, 0.98,
            "Quadrature sum = $(I_hat)",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left"
        )
    end

    # ------------------------------------------------------------
    # Compute how many blocks cover each node index j (0..N).
    # Shared nodes (block boundaries) will have cover_count > 1.
    # We'll split contrib[j] evenly across covering blocks.
    # ------------------------------------------------------------
    cover_count = zeros(Int, length(xs))  # index = j+1

    function _accumulate_cover!(start::Int, kind::Symbol)
        w = _local_width(p, kind)
        jlo = max(start, 0)
        jhi = min(start + w, N)
        for j in jlo:jhi
            cover_count[j+1] += 1
        end
    end

    # One pass over the tiling, same as the drawing order
    start_tmp = 0

    # Left block
    if Ltype === :closed
        _accumulate_cover!(start_tmp, :closed)
        start_tmp += (p - 1)
    else
        _accumulate_cover!(start_tmp, :opened)
        start_tmp += p
    end

    # Interior closed blocks
    for _ in 1:m
        _accumulate_cover!(start_tmp, :closed)
        start_tmp += (p - 1)
    end

    # Right block
    start_expected = N - wR
    start_tmp == start_expected || JobLoggerTools.error_benji(
        "Internal tiling mismatch in cover_count pass: start=$start_tmp expected=$start_expected"
    )

    if Rtype === :closed
        _accumulate_cover!(start_tmp, :closed)
    else
        _accumulate_cover!(start_tmp, :opened)
    end

    # -------------------------------
    # Composite blocks: left, interior (closed), right
    # Each block visualized by filling under its interpolant
    # -------------------------------
    start = 0

    function _block_nodes_indices(start::Int, kind::Symbol, which_open::Symbol)::Vector{Int}
        if kind === :closed
            # u = 0:(p-1)
            return [start + u for u in 0:(p-1)]
        else
            if which_open === :backward
                # u = 1:p
                return [start + u for u in 1:p]
            elseif which_open === :forward
                # u = 0:(p-1)
                return [start + u for u in 0:(p-1)]
            else
                JobLoggerTools.error_benji("which_open must be :backward or :forward")
            end
        end
    end

    function _block_interval_x(start::Int, kind::Symbol)::Tuple{Float64,Float64}
        w = _local_width(p, kind)
        x0 = aa + start * h
        x1 = aa + (start + w) * h
        return x0, x1
    end

    function _draw_block(start::Int, kind::Symbol, which_open::Symbol)
        inds = _block_nodes_indices(start, kind, which_open)
        (minimum(inds) >= 0 && maximum(inds) <= N) || return

        xv = [xs[j+1] for j in inds]
        yv = Vector{Float64}(undef, length(xv))
        @inbounds for i in eachindex(xv)
            yi = f(xv[i])
            yv[i] = (isfinite(yi) ? Float64(yi) : NaN)
        end
        any(!isfinite, yv) && return

        c = _poly_fit_coeffs(xv, yv)

        x0, x1 = _block_interval_x(start, kind)
        xb = collect(range(x0, x1; length=ngrid_block))
        yb = Vector{Float64}(undef, length(xb))
        @inbounds for i in eachindex(xb)
            yb[i] = _poly_eval(c, xb[i])
        end

        # --- Let matplotlib choose the next color automatically ---
        line = ax.plot(xb, yb; linewidth=1.2, alpha=0.85)[1]
        col = line."get_color"()   # <- exact color used for this block line

        if show_block_fill
            ax.fill_between(xb, zeros(length(xb)), yb; alpha=0.18, linewidth=0, color=col)
        end

        # # --- Color bars in this block with the same color ---
        # # Split shared-node contributions evenly across covering blocks.
        # if show_bars
        #     w = _local_width(p, kind)
        #     jlo = max(start, 0)
        #     jhi = min(start + w, N)

        #     js = Int[]
        #     hs_local = Float64[]

        #     for j in jlo:jhi
        #         idx = j + 1
        #         if isfinite(contrib[idx]) && cover_count[idx] > 0
        #             # split contribution for shared nodes
        #             csplit = contrib[idx] / cover_count[idx]
        #             push!(js, j)
        #             push!(hs_local, csplit / bw)
        #         end
        #     end

        #     if !isempty(js)
        #         ax.bar(
        #             [xs[j+1] for j in js],
        #             hs_local;
        #             width=bw,
        #             align="center",
        #             alpha=0.35,
        #             linewidth=0,
        #             color=col
        #         )
        #     end
        # end

        # ============================================================
        # Exact, non-distorting visualization:
        # Fill the area under each nodal contribution y_i * ℓ_i(x)
        # so that the sum equals the interpolant P(x) exactly.
        # Works for all ns_pK, any boundary tiling.
        # ============================================================
        if show_bars
            Lmat = _lagrange_basis_matrix(xv, xb)   # size (p, ngrid_block)

            # We draw per-node patches with outlines, so the splitting is visible.
            # Positive and negative parts are both drawn (no hiding).
            for i in 1:length(xv)
                yi = yv[i]
                # contribution curve over this block grid
                ycomp = yi .* view(Lmat, i, :)

                # fill between 0 and ycomp (signed)
                ax.fill_between(
                    xb,
                    zeros(length(xb)),
                    ycomp;
                    facecolor=col,
                    alpha=0.25,
                    edgecolor="black",
                    linewidth=0.6
                )
            end
        end

        return
    end

    bidx = 0  # block index for color cycling

    # Left block
    if Ltype === :closed
        bidx += 1
        _draw_block(start, :closed, :backward)
        start += (p - 1)
    else
        bidx += 1
        _draw_block(start, :opened, :backward)
        start += p
    end

    # Interior closed blocks
    for _ in 1:m
        bidx += 1
        _draw_block(start, :closed, :backward)
        start += (p - 1)
    end

    # Right block (start should be N - wR)
    start_expected = N - wR
    start == start_expected || JobLoggerTools.error_benji("Internal tiling mismatch: start=$start expected=$start_expected")

    if Rtype === :closed
        bidx += 1
        _draw_block(start, :closed, :backward)
    else
        bidx += 1
        _draw_block(start, :opened, :forward)
    end

    ax.set_xlim(aa, bb)
    ax.set_xlabel(raw"$x$")
    ax.set_ylabel(raw"$f(x)$ / block interpolants")

    ax.set_title("Rule coverage: rule=$(String(rule)), boundary=$(String(boundary)), N=$N")

    fig.tight_layout()
    outfile = "quad_coverage_$(name)_$(String(rule))_$(String(boundary))_N$(N).png"
    fig.savefig(outfile)
    PyPlot.close(fig)

    return nothing
end

end  # module PlotTools