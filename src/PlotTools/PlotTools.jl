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
        rule::Symbol = :ns_p3,
        boundary::Symbol = :LCRC
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

## Output

The output file is saved as:

```julia
convergence_\$(name)_\$(rule)_\$(boundary).png
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

Visualize **``1``-dimensional quadrature coverage** on ``[a,b]`` by plotting the integrand,
the quadrature nodes/weights, and an **exact, non-distorting decomposition**
of the per-block interpolation contributions.

# Function description

This routine is a diagnostic/educational visualizer for understanding what a
``1``-dimensional quadrature rule is *actually doing* on a structured grid.

Given a rule specified by `(rule, boundary)` and a subdivision count `N`,
the function:

1. Retrieves the global quadrature nodes/weights `(xs, ws)` via
   [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref).
2. Partitions the domain into a sequence of **blocks** (local panels), each associated
   with a set of nodes used to form a local interpolant.
3. For each block, constructs an interpolating polynomial through the nodal samples
   ``(x_i, f(x_i))`` using a simple Vandermonde solve (educational; not optimized).
4. Draws the block interpolant curve and, optionally, fills the signed contribution
   of each nodal term **exactly** using Lagrange basis functions.

The goal is to provide an *honest* visualization:

- The filled contributions in a block sum to the block interpolant exactly.
- Both positive and negative contributions are shown (no clipping / hiding).

# Block construction (rule-family dispatch)

This plotter supports three rule families, identified by `rule`:

- Newton-Cotes NS rules (as detected by [`Maranatha.Quadrature.NewtonCotes._is_ns_rule`](@ref))
- Gauss-family rules (as detected by [`Maranatha.Quadrature.Gauss._is_gauss_rule(rule)`](@ref))
- B-spline-family rules (as detected by [`Maranatha.Quadrature.BSpline._is_bspl_rule(rule)`](@ref))

## NS rules (`:ns_pK`)

For NS rules, blocks are constructed using the same boundary-aware tiling rules
as the composite quadrature assembly. The boundary pattern `boundary` must be one of:

- `:LCRC`, `:LORC`, `:LCRO`, `:LORO`

and `N` must satisfy the composability constraint imposed by the boundary widths.

## Gauss / B-spline rules

For Gauss-family and B-spline-family rules, the visualization uses a minimal-assumption
panelization: the interval is divided into `N` uniform panels, and each global node is
assigned to a panel based on its `x` location. Empty panels are skipped.

This is intended as a conservative, non-distorting *coverage* visualization for
structured composite constructions.

# Exact, non-distorting contribution visualization

When `show_bars=true`, the routine performs a per-block decomposition based on
Lagrange basis functions.

For a block with nodes ``x_1,\\dots,x_p`` and nodal values ``y_i = f(x_i)``,
the interpolant is:

```math
P(x) = \\sum_{i=1}^{p} y_i \\, \\ell_i(x),
```

where ``\\ell_i(x)`` are Lagrange basis polynomials.

The visualization fills the signed areas under each term ``y_i\\,\\ell_i(x)``
over the block grid. Therefore, the sum of filled regions equals the plotted
block interpolant exactly.

Implementation notes:

* Lagrange basis values are evaluated using a barycentric form for stability.
* The block interpolant itself is computed from a Vandermonde fit to keep the
  educational mapping between nodal samples and the interpolant explicit.

# Arguments

* `f`:
  Scalar integrand ``f(x)`` defined on ``[a,b]``. Non-finite samples are treated as `NaN`
  and will suppress drawing in blocks where needed.

* `a`, `b`:
  Interval endpoints.

* `N::Int`:
  Subdivision count that defines the structured grid resolution and (for composite rules)
  the panelization used by [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref).

# Keyword arguments

* `rule::Symbol = :ns_p3`:
  Quadrature rule identifier. Must be supported by the internal rule-family detectors:
  [`Maranatha.Quadrature.NewtonCotes._is_ns_rule`](@ref), 
  [`Maranatha.Quadrature.Gauss._is_gauss_rule`](@ref), or 
  [`Maranatha.Quadrature.BSpline._is_bspl_rule`](@ref).

* `boundary::Symbol = :LCRC`:
  Boundary pattern. Used by NS rules for exact boundary tiling; for other families it is
  forwarded to node/weight generation (and may be ignored depending on the backend).

* `ngrid_f::Int = 4000`:
  Number of points used to draw the dense integrand curve.

* `ngrid_block::Int = 400`:
  Number of points used per block for interpolant curves and basis contribution fills.

* `name::String = "coverage"`:
  Label used in the output filename.

* `show_bars::Bool = true`:
  If `true`, draw the exact Lagrange-term contribution fills inside each block.

* `bar_width_scale::Float64 = 1.0`:
  Width scale used only for the legacy bar-height bookkeeping and diagnostic `I_hat`
  display (the final visualization uses the Lagrange-term fills; no geometric distortion
  is introduced by this parameter).

* `show_block_fill::Bool = false`:
  If `true`, also fill the region under each block interpolant curve (in addition to the
  per-term contribution fills). This is purely visual.

# Output

Saves a PNG file:

```julia
quad_coverage_\$(name)_\$(String(rule))_\$(String(boundary))_N\$(N).png
```

# Returns

* `nothing`.

# Errors

* Throws an error if `N < 1`.
* Throws an error if `rule` does not belong to a supported family.
* For NS rules, throws an error if `boundary` is invalid or `N` violates the composability
  constraint implied by `(rule, boundary)`.

# Design notes

* This routine is intended for **diagnostics and interpretation**, not for performance.
* The interpolant fit uses a Vandermonde solve for transparency; it is not numerically optimal.
* The contribution visualization is designed to avoid “curve-smoothing” or other presentation
  artifacts that could misrepresent what the quadrature rule is actually approximating.
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

    # Identify rule family (must match your existing modules)
    is_ns   = Quadrature.NewtonCotes._is_ns_rule(rule)
    is_gaus = Quadrature.Gauss._is_gauss_rule(rule)
    is_bs   = Quadrature.BSpline._is_bspl_rule(rule)

    (is_ns || is_gaus || is_bs) || JobLoggerTools.error_benji(
        "Unsupported rule family for coverage plot: rule=$rule"
    )

    # -------------------------------
    # Build global nodes/weights (markers + contributions)
    # -------------------------------
    xs, ws = Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)

    aa = Float64(a)
    bb = Float64(b)
    h  = (bb - aa) / Float64(N)

    # -------------------------------
    # Build "blocks" for honest visualization
    #
    # Each block is a NamedTuple:
    #   (x0::Float64, x1::Float64, inds::Vector{Int})
    #
    # - For NS rules: use exact boundary tiling (same as your current logic).
    # - For Gauss / B-spline: treat as N uniform panels on [a,b] and assign nodes by location.
    #   (This is honest, minimal-assumption, and works as long as your composite construction
    #    respects the N-panel partition.)
    # -------------------------------
    blocks = NamedTuple{(:x0,:x1,:inds),Tuple{Float64,Float64,Vector{Int}}}[]

    if is_ns
        # --- NS: keep your exact tiling logic, but encapsulate as block list ---

        # decode boundary
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

        # NS order p from :ns_pK
        startswith(String(rule), "ns_p") || JobLoggerTools.error_benji("NS rule must be :ns_pK (got rule=$rule)")
        p = parse(Int, String(rule)[5:end])
        p >= 2 || JobLoggerTools.error_benji("p must be ≥ 2 (got p=$p from rule=$rule)")

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
            return m, wL, wR, Ltype, Rtype, p
        end

        m, wL, wR, Ltype, Rtype, p = _check_condition(p, boundary, N)

        # helper to get block node indices (node index j is 0..N; stored at j+1)
        function _block_nodes_indices(start::Int, kind::Symbol, which_open::Symbol)::Vector{Int}
            if kind === :closed
                return [start + u for u in 0:(p-1)]
            else
                if which_open === :backward
                    return [start + u for u in 1:p]
                elseif which_open === :forward
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

        # build blocks in drawing order: left, interior, right
        start = 0

        # left
        if Ltype === :closed
            x0, x1 = _block_interval_x(start, :closed)
            inds = _block_nodes_indices(start, :closed, :backward)
            push!(blocks, (; x0, x1, inds))
            start += (p - 1)
        else
            x0, x1 = _block_interval_x(start, :opened)
            inds = _block_nodes_indices(start, :opened, :backward)
            push!(blocks, (; x0, x1, inds))
            start += p
        end

        # interior closed
        for _ in 1:m
            x0, x1 = _block_interval_x(start, :closed)
            inds = _block_nodes_indices(start, :closed, :backward)
            push!(blocks, (; x0, x1, inds))
            start += (p - 1)
        end

        # right
        start_expected = N - wR
        start == start_expected || JobLoggerTools.error_benji("Internal tiling mismatch: start=$start expected=$start_expected")

        if Rtype === :closed
            x0, x1 = _block_interval_x(start, :closed)
            inds = _block_nodes_indices(start, :closed, :backward)
            push!(blocks, (; x0, x1, inds))
        else
            x0, x1 = _block_interval_x(start, :opened)
            inds = _block_nodes_indices(start, :opened, :forward)
            push!(blocks, (; x0, x1, inds))
        end

    else
        # --- Gauss / B-spline: N uniform panels on [a,b], assign nodes to panels by x location ---

        edges = collect(range(aa, bb; length=N+1))

        # panel index in 1..N (clamp to handle x==bb)
        @inline function _panel_id(x::Float64)::Int
            if x <= edges[1]
                return 1
            elseif x >= edges[end]
                return N
            else
                # find k s.t. edges[k] <= x < edges[k+1]
                return searchsortedlast(edges, x) |> xk -> min(max(xk, 1), N)
            end
        end

        panel_inds = [Int[] for _ in 1:N]
        @inbounds for j in eachindex(xs)
            pid = _panel_id(Float64(xs[j]))
            push!(panel_inds[pid], j)  # NOTE: j is 1-based index into xs/ws
        end

        for k in 1:N
            x0 = edges[k]
            x1 = edges[k+1]
            inds = panel_inds[k]
            # Skip empty panels (can happen for some constructions)
            isempty(inds) && continue
            push!(blocks, (; x0, x1, inds))
        end
    end

    # m, wL, wR, Ltype, Rtype = _check_condition(p, boundary, N)

    # -------------------------------
    # Build global nodes/weights (for markers)
    # -------------------------------
    xs, ws = Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)

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

    # -------------------------------
    # Draw blocks (backend-agnostic)
    # Each block: (x0, x1, inds)
    # -------------------------------
    function _draw_block_generic(x0::Float64, x1::Float64, inds_any)
        # inds_any:
        # - NS path: node indices are j = 0..N (global grid index)
        # - Gauss/BS path: node indices are direct 1-based indices into xs/ws
        #
        # Normalize to actual x-node vectors xv
        xv = Float64[]
        if is_ns
            # inds are in 0..N => map to xs[j+1]
            append!(xv, [Float64(xs[j+1]) for j in inds_any])
        else
            # inds are 1-based into xs
            append!(xv, [Float64(xs[j]) for j in inds_any])
        end

        # y at nodes
        yv = Vector{Float64}(undef, length(xv))
        @inbounds for i in eachindex(xv)
            yi = f(xv[i])
            yv[i] = isfinite(yi) ? Float64(yi) : NaN
        end
        any(!isfinite, yv) && return

        # Fit interpolant (educational Vandermonde)
        c = _poly_fit_coeffs(xv, yv)

        xb = collect(range(x0, x1; length=ngrid_block))
        yb = Vector{Float64}(undef, length(xb))
        @inbounds for i in eachindex(xb)
            yb[i] = _poly_eval(c, xb[i])
        end

        # Let matplotlib choose color for this block
        line = ax.plot(xb, yb; linewidth=1.2, alpha=0.85)[1]
        col  = line."get_color"()

        if show_block_fill
            ax.fill_between(xb, zeros(length(xb)), yb; alpha=0.18, linewidth=0, color=col)
        end

        # ============================================================
        # Exact, non-distorting visualization:
        # Fill the area under each nodal contribution y_i * ℓ_i(x)
        # so that the sum equals the interpolant P(x) exactly.
        # ============================================================
        if show_bars
            Lmat = _lagrange_basis_matrix(xv, xb)
            for i in 1:length(xv)
                ycomp = yv[i] .* view(Lmat, i, :)
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

    for blk in blocks
        _draw_block_generic(blk.x0, blk.x1, blk.inds)
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