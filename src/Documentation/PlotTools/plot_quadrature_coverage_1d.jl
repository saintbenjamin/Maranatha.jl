# ============================================================================
# src/Documentation/PlotTools/plot_quadrature_coverage_1d.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

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
        save_file::Bool = false,
    ) -> Nothing

Visualize one-dimensional quadrature behavior on `[a,b]` using a pedagogical plot
that overlays the sampled integrand with a rule-family-specific area interpretation
of the quadrature construction.

# Arguments
- `f`:
  Scalar integrand to be sampled by the selected 1D quadrature rule.
- `a::Real`, `b::Real`:
  Interval endpoints.
- `N::Int`:
  Subdivision count forwarded to the quadrature backend.

# Keyword arguments
- `rule::Symbol = :newton_p3`, `boundary::Symbol = :LU_ININ`:
  Rule selector forwarded to the quadrature backend.
- `ngrid_f::Int = 4000`:
  Number of points used to draw the dense reference curve of `f`.
- `ngrid_block::Int = 400`:
  Number of points used per local block / span in rule-specific visualizations.
- `name::String = "demo"`:
  Basename used for output filenames.
- `figs_dir::String = "."`:
  Output directory for saved figures.
- `save_file::Bool = false`:
  If `true`, save the generated figure.

# Returns
- `Nothing`:
  This routine is used for plotting and optional file-output side effects.

# Errors
- Throws an error if `N < 1`, if the interval is invalid, or if the rule family is unsupported.
- Propagates backend errors from quadrature-node construction and plotting.
- In the B-spline branch, may throw if node samples are not finite.

# Notes
- This routine is intended for intuition-building, inspection, and debugging of 1D rules.
- In the B-spline branch, the routine reconstructs the effective spline curve from the
  sampled node values and fills its contribution span by span.
- In the non-B-spline branch, the routine uses an educational width=`w_i`,
  height=`f(x_i)` area view arranged sequentially from `a`; if a weight is
  negative, the local bar orientation is flipped so the signed area remains
  `w_i f(x_i)`.
- A [`TOML`](https://toml.io/en/)-driven convenience wrapper is also provided.
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

    T = promote_type(typeof(a), typeof(b))

    aa = convert(T, a)
    bb = convert(T, b)
    (isfinite(aa) && isfinite(bb)) || JobLoggerTools.error_benji("a,b must be finite (got a=$a, b=$b)")
    (bb > aa) || JobLoggerTools.error_benji("Require b > a (got a=$a, b=$b)")

    # Identify rule family (must match your existing modules)
    is_ns   = NewtonCotes._is_newton_cotes_rule(rule)
    is_gaus = Gauss._is_gauss_rule(rule)
    is_bs   = BSpline._is_bspline_rule(rule)
    (is_ns || is_gaus || is_bs) || JobLoggerTools.error_benji("Unsupported rule family: rule=$rule")

    # Build global nodes/weights (single source of truth)
    xs, ws = QuadratureNodes.get_quadrature_1d_nodes_weights(a, b, N, rule, boundary)
    (length(xs) == length(ws)) || JobLoggerTools.error_benji("Internal: xs/ws length mismatch")

    # -------------------------------
    # Helpers: safe f eval
    # -------------------------------
    @inline function _f_val(x)
        y = f(x)
        return (y isa Real && isfinite(y)) ? convert(typeof(x), y) : convert(typeof(x), NaN)
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
        (length(xs) == nb) || JobLoggerTools.error_benji(
            "BSpline internal mismatch: length(xs)=$(length(xs)) nb=$nb"
        )

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
    yg = Vector{T}(undef, length(xg))
    @inbounds for i in eachindex(xg)
        yg[i] = _f_val(xg[i])
    end
    ax.plot(xg, yg; linewidth=2.2)

    # Node samples (for scatter + for quadrature sum)
    y_nodes = Vector{T}(undef, length(xs))
    @inbounds for j in eachindex(xs)
        y_nodes[j] = _f_val(convert(T, xs[j]))
    end

    # Quadrature sum text
    I_hat = zero(T)
    @inbounds for j in eachindex(xs)
        yj = y_nodes[j]
        wj = convert(T, ws[j])
        if isfinite(yj) && isfinite(wj)
            I_hat += wj * yj
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
        # - Applies to non-BSpline rules here.
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

"""
    plot_quadrature_coverage_1d(
        toml_path::AbstractString;
        N::Union{Nothing,Int} = nothing,
        ngrid_f::Int = 4000,
        ngrid_block::Int = 400,
        name::Union{Nothing,String} = nothing,
        figs_dir::Union{Nothing,String} = nothing,
        save_file::Bool = false,
    ) -> Nothing

Convenience wrapper that loads a Maranatha [`TOML`](https://toml.io/en/) configuration, imports the user
integrand, and forwards the recovered inputs to the primary
`plot_quadrature_coverage_1d` method.

# Arguments
- `toml_path::AbstractString`:
  Path to the [`TOML`](https://toml.io/en/) configuration file.

# Keyword arguments
- `N::Union{Nothing,Int} = nothing`:
  Subdivision count to plot. If omitted, all configured `nsamples` are used.
- `ngrid_f::Int = 4000`:
  Number of dense plotting points for the true integrand curve.
- `ngrid_block::Int = 400`:
  Number of dense plotting points per local block / span.
- `name::Union{Nothing,String} = nothing`:
  Optional basename overriding the [`TOML`](https://toml.io/en/) configuration.
- `figs_dir::Union{Nothing,String} = nothing`:
  Optional output directory overriding the [`TOML`](https://toml.io/en/) configuration.
- `save_file::Bool = false`:
  If `true`, save the generated figures.

# Returns
- `Nothing`.

# Errors
- Propagates [`TOML`](https://toml.io/en/) parsing, validation, and integrand-loading errors.
- Throws an error if an explicitly requested `N` is not contained in `cfg.nsamples`.
- Propagates all plotting errors from the primary method.

# Notes
- Because the integrand is loaded dynamically, this wrapper dispatches through `Base.invokelatest`.
"""
function plot_quadrature_coverage_1d(
    toml_path::AbstractString;
    N::Union{Nothing,Int} = nothing,
    ngrid_f::Int = 4000,
    ngrid_block::Int = 400,
    name::Union{Nothing,String} = nothing,
    figs_dir::Union{Nothing,String} = nothing,
    save_file::Bool = false,
)::Nothing
    cfg = MaranathaTOML.parse_run_config_from_toml(toml_path)
    MaranathaTOML.validate_run_config(cfg)

    (cfg.dim == 1) || JobLoggerTools.error_benji(
        "plot_quadrature_coverage_1d supports only dim = 1 (got dim=$(cfg.dim))"
    )

    integrand = MaranathaTOML.load_integrand_from_file(
        cfg.integrand_file;
        func_name = cfg.integrand_name
    )

    plot_name = isnothing(name) ? cfg.name_prefix : name
    plot_dir  = isnothing(figs_dir) ? cfg.save_path : figs_dir

    Ns = if isnothing(N)
        collect(Int.(cfg.nsamples))
    else
        N in cfg.nsamples || error(
            "Requested N=$N is not present in TOML nsamples = $(cfg.nsamples)."
        )
        [N]
    end

    for Ni in Ns
        Base.invokelatest(
            plot_quadrature_coverage_1d,
            integrand,
            cfg.a,
            cfg.b,
            Ni;
            rule = cfg.rule,
            boundary = cfg.boundary,
            ngrid_f = ngrid_f,
            ngrid_block = ngrid_block,
            name = plot_name,
            figs_dir = plot_dir,
            save_file = save_file,
        )
    end

    return nothing
end