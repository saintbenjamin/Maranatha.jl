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

Visualize **``1``-dimensional quadrature behavior** on ``[a,b]`` by plotting the true
integrand ``f(x)`` together with a **pedagogical representation** of how the selected
quadrature rule contributes to the integral.

Unlike [`Maranatha.PlotTools.plot_convergence_result`](@ref), this routine is not a
fit-visualization tool. Instead, it is meant to help interpret the geometric or
rule-structural meaning of a selected 1D quadrature construction.

# What this routine draws

This routine is primarily intended for inspection, intuition-building, and debugging of
1D quadrature rules. It answers questions such as:

* what nodes and weights the backend is using,
* what effective curve is being integrated in the B-spline case,
* and how signed contributions are being assembled in non-B-spline rules.

This function always draws:

1. A dense curve of the true integrand ``f(x)`` over ``[a,b]``.
2. Quadrature nodes/weights ``(xs, ws)`` obtained from
   [`Maranatha.Quadrature.QuadratureDispatch.get_quadrature_1d_nodes_weights`](@ref).
3. A text annotation displaying the quadrature sum computed directly from the returned
   nodes and weights.

   In the current implementation, this is displayed as a plain text label summarizing
   the numerical quadrature sum, rather than as a separately typeset mathematical object.

Rather than being fixed at a hard-coded corner, this annotation is placed using
[`_smart_text_placement!`](@ref), an internal plotting helper that heuristically
searches candidate locations and tries to avoid overlap with plotted curves and
sampled quadrature information.

Then it draws **one** of the following rule-specific visualizations:

## 1) B-spline rules (`is_bs == true`)

For B-spline quadrature rules, this plotter reconstructs the **effective spline curve**
implicitly used by the B-spline backend and fills its area:

* Sample data: `y_j = f(x_j)` at Greville nodes.
* Reconstruct spline coefficients by solving the same collocation system used by the rule.
* Plot the spline curve **piecewise per knot span** and fill the region under each span.
  (Each span gets its own color, and the fill color is matched to the span line color.)

This visualization is meant to show *which effective curve the B-spline backend is integrating*.
The reconstructed spline curve is also passed to the smart annotation-placement helper
so that the quadrature-sum label avoids covering the visually relevant spline shape.

## 2) Non-B-spline rules (`is_bs == false`)

For Newton-Cotes and Gauss-family rules, this plotter uses a simple **mass-bar view**:

This view is intentionally schematic: it is designed to visualize signed quadrature
contributions clearly, not to reproduce the original node geometry exactly.

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

* `f`:
  Scalar integrand to be sampled by the selected 1D quadrature rule.
  If ``f(x)`` is not finite at a node, that node is skipped when accumulating the
  displayed quadrature sum ``\\hat I``.
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

# Typical use cases

This routine is especially useful when:

1. checking how a 1D rule samples the integrand,
2. inspecting the effective reconstructed curve for B-spline quadrature,
3. illustrating signed weight contributions in Newton-Cotes or Gauss-family rules,
4. preparing pedagogical figures for notes, talks, or debugging workflows

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
Only a single pedagogical figure is generated by this routine.

# Returns

`nothing`.

This routine is used for its side effects: it displays the generated figure and,
if `save_file = true`, also writes it to disk.

# Example

The example below generates a pedagogical 1D coverage plot for a Gauss-family rule.

```julia
f(x) = sin(3x) * exp(-x^2)

plot_quadrature_coverage_1d(
    f,
    0.0,
    1.0,
    6;
    rule = :gauss_p4,
    boundary = :LU_EXEX,
    name = "gauss_demo",
    save_file = false
)
```
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
    is_ns   = NewtonCotes._is_newton_cotes_rule(rule)
    is_gaus = Gauss._is_gauss_rule(rule)
    is_bs   = BSpline._is_bspline_rule(rule)
    (is_ns || is_gaus || is_bs) || JobLoggerTools.error_benji("Unsupported rule family: rule=$rule")

    # Build global nodes/weights (single source of truth)
    xs, ws = QuadratureDispatch.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)
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
        p    = BSpline._parse_bspline_p(rule)
        kind = BSpline._bspline_kind(rule)  # :interp or :smooth

        t  = BSpline._build_knots_uniform(a, b, N, p, boundary)
        nb = length(t) - p - 1
        (length(xs) == nb) || JobLoggerTools.error_benji("BSpline internal mismatch: length(xs)=$(length(xs)) nb=$nb")

        # Build A[j,i] = B_i(xs[j])
        A = Matrix{Float64}(undef, nb, nb)
        @inbounds for j in 1:nb
            Bj = BSpline._bspline_basis_all(xs[j], t, p)
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
            R = BSpline._roughness_R_second_diff(nb)
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
        B = BSpline._bspline_basis_all(x, t, p)
        return LinearAlgebra.dot(c, B)
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