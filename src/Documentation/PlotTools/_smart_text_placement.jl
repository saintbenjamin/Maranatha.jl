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
  Optional polyline coordinates to avoid.
- `yerr_points::Union{Nothing,AbstractVector{<:Real}} = nothing`:
  Optional vertical error magnitudes associated with `x_points` / `y_points`.
- `fontsize::Real = 10`:
  Font size used for the annotation box.
- `prefer_order`:
  Candidate anchor positions, given in axis-fraction coordinates.

# Returns
- `Nothing`:
  The function mutates `ax` by adding the selected annotation.

# Errors
- Propagates plotting-backend errors if renderer information cannot be obtained.
- May propagate dimension or indexing errors if the supplied coordinate vectors are inconsistent.

# Notes
- This is an internal plotting helper.
- Placement is evaluated in display space after rendering, so the chosen location
  depends on the actual rendered axis geometry.
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