# ============================================================================
# src/Documentation/PlotTools/_smart_text_placement!.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _smart_text_placement!(
        fig,
        ax;
        text::AbstractString,
        x_points::AbstractVector{<:Real},
        y_points::AbstractVector{<:Real},
        x_curve::AbstractVector{<:Real}=Real[],
        y_curve::AbstractVector{<:Real}=Real[],
        yerr_points::Union{Nothing,AbstractVector{<:Real}}=nothing,
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

Place an annotation text box inside a plot axis while heuristically avoiding
plotted points, curves, and vertical error bars.

# Function description
This helper evaluates a list of candidate text-anchor positions in axis-fraction
coordinates, temporarily renders an invisible text box at each candidate, and
scores the resulting display-space bounding box against nearby plot content.

The score penalizes:

- overlap with the axis frame,
- direct overlap with plotted data points,
- direct overlap with vertical error-bar segments,
- direct overlap with plotted curve segments, and
- near-misses within a fixed display-space padding window.

After all candidate anchors in `prefer_order` are evaluated, the best-scoring
placement is chosen and the visible annotation box is added to `ax`.

# Arguments
- `fig`:
  Figure object used to access the renderer.
- `ax`:
  Axis object on which the text is placed.
- `text::AbstractString`:
  Annotation string to place.
- `x_points::AbstractVector{<:Real}`, `y_points::AbstractVector{<:Real}`:
  Data points to avoid when selecting the text position.
- `x_curve::AbstractVector{<:Real} = Real[]`, `y_curve::AbstractVector{<:Real} = Real[]`:
  Optional polyline coordinates to avoid. These must either both be empty or
  both be non-empty.
- `yerr_points::Union{Nothing,AbstractVector{<:Real}} = nothing`:
  Optional vertical error magnitudes associated with `x_points` / `y_points`.
- `fontsize::Real = 10`:
  Font size used for the annotation box.
- `prefer_order`:
  Candidate anchor positions, given as tuples
  `(x_fraction, y_fraction, vertical_align, horizontal_align)` in axis-fraction
  coordinates.

# Returns
- `Nothing`:
  The function mutates `ax` by adding the selected annotation.

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if
  `length(x_points) != length(y_points)`.
- Throws if `yerr_points !== nothing` but `length(yerr_points) != length(x_points)`.
- Throws if exactly one of `x_curve` and `y_curve` is empty.
- Throws if both curve arrays are present but their lengths differ.
- Propagates plotting-backend errors if renderer information cannot be obtained.

# Notes
- This is an internal plotting helper.
- Placement is evaluated in display space after rendering, so the chosen location
  depends on the actual rendered axis geometry.
- Error bars are treated as vertical line segments centered on `y_points`.
- Curve avoidance uses consecutive pairs of `(x_curve, y_curve)` as polyline
  segments.
- If no candidate survives scoring logic, the helper falls back to the
  upper-left corner anchor `(0.02, 0.98, :top, :left)`.
"""
function _smart_text_placement!(
    fig,
    ax;
    text::AbstractString,
    x_points::AbstractVector{<:Real},
    y_points::AbstractVector{<:Real},
    x_curve::AbstractVector{<:Real}=Real[],
    y_curve::AbstractVector{<:Real}=Real[],
    yerr_points::Union{Nothing,AbstractVector{<:Real}}=nothing,
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
    length(x_points) == length(y_points) || JobLoggerTools.error_benji(
        "Length mismatch in _smart_text_placement!: length(x_points) != length(y_points)"
    )

    if yerr_points !== nothing
        length(yerr_points) == length(x_points) || JobLoggerTools.error_benji(
            "Length mismatch in _smart_text_placement!: length(yerr_points) != length(x_points)"
        )
    end

    isempty(x_curve) == isempty(y_curve) || JobLoggerTools.error_benji(
        "Curve inputs must both be empty or both be non-empty in _smart_text_placement!."
    )

    if !isempty(x_curve)
        length(x_curve) == length(y_curve) || JobLoggerTools.error_benji(
            "Length mismatch in _smart_text_placement!: length(x_curve) != length(y_curve)"
        )
    end

    x_points_f = Float64.(x_points)
    y_points_f = Float64.(y_points)
    x_curve_f  = Float64.(x_curve)
    y_curve_f  = Float64.(y_curve)
    yerr_f = yerr_points === nothing ? nothing : Float64.(yerr_points)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

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
        if _point_inside_bbox(p1x, p1y, x0, x1, y0, y1) ||
           _point_inside_bbox(p2x, p2y, x0, x1, y0, y1)
            return true
        end

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

    pts = isempty(x_points_f) ?
        zeros(0, 2) :
        ax.transData.transform(hcat(x_points_f, y_points_f))

    err_segments = Tuple{Float64,Float64,Float64,Float64}[]
    if yerr_f !== nothing && !isempty(x_points_f)
        ylo = y_points_f .- yerr_f
        yhi = y_points_f .+ yerr_f
        p_lo = ax.transData.transform(hcat(x_points_f, ylo))
        p_hi = ax.transData.transform(hcat(x_points_f, yhi))
        for i in eachindex(x_points_f)
            push!(err_segments, (p_lo[i,1], p_lo[i,2], p_hi[i,1], p_hi[i,2]))
        end
    end

    crv_segments = Tuple{Float64,Float64,Float64,Float64}[]
    if !isempty(x_curve_f) && length(x_curve_f) >= 2
        crv = ax.transData.transform(hcat(x_curve_f, y_curve_f))
        for i in 1:(size(crv,1)-1)
            push!(crv_segments, (crv[i,1], crv[i,2], crv[i+1,1], crv[i+1,2]))
        end
    end

    axbb = ax.get_window_extent(renderer=renderer)

    best = nothing
    best_score = Inf

    pad = 8.0
    near_pad = 18.0

    for (xf, yf, va_sym, ha_sym) in prefer_order
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

        if x0 < axbb.x0 || x1 > axbb.x1 || y0 < axbb.y0 || y1 > axbb.y1
            score += 1e6
        end

        @inbounds for i in axes(pts, 1)
            px = pts[i, 1]
            py = pts[i, 2]

            if _point_inside_bbox(px, py, x0, x1, y0, y1)
                score += 500.0
            else
                d = _point_rect_distance(px, py, x0, x1, y0, y1)
                if d < near_pad
                    score += 20.0 * (1.0 - d / near_pad)
                end
            end
        end

        for (xA, yA, xB, yB) in err_segments
            d = _segment_rect_distance(xA, yA, xB, yB, x0, x1, y0, y1)
            if d == 0.0
                score += 1200.0
            elseif d < near_pad
                score += 60.0 * (1.0 - d / near_pad)
            end
        end

        for (xA, yA, xB, yB) in crv_segments
            d = _segment_rect_distance(xA, yA, xB, yB, x0, x1, y0, y1)
            if d == 0.0
                score += 120.0
            elseif d < near_pad
                score += 8.0 * (1.0 - d / near_pad)
            end
        end

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