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

Internal plotting helper that places a text box inside an axis while heuristically
avoiding plotted data, fitted curves, and vertical error bars.

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