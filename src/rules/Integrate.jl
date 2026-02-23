module Integrate

using ..Simpson13Rule, ..Simpson38Rule, ..BodeRule
using ..Simpson13Rule_MinOpen_MaxOpen, ..Simpson38Rule_MinOpen_MaxOpen, ..BodeRule_MinOpen_MaxOpen

export integrate_nd, quadrature_1d_nodes_weights

"""
    integrate_nd(integrand, a, b, N, dim, rule)

Dispatch integration method by dimension and rule.

# Arguments
- `integrand`: function of 1 to 4 variables
- `a`, `b`: integration bounds (same for all dimensions)
- `N`: number of intervals
- `dim`: dimensionality (1, 2, 3, or 4)
- `rule`: integration rule symbol (e.g. `:simpson13_close`, `:simpson38_close`, `:bode_close`)

# Returns
- Estimated integral value (Float64)
"""
function integrate_nd(integrand, a, b, N, dim, rule)
    if dim == 1
        return integrate_1d(integrand, a, b, N, rule)
    elseif dim == 2
        return integrate_2d(integrand, a, b, N, rule)
    elseif dim == 3
        return integrate_3d(integrand, a, b, N, rule)
    elseif dim == 4
        return integrate_4d(integrand, a, b, N, rule)
    else
        error("Dimension $dim not supported. Use dim = 1, 2, 3, or 4.")
    end
end

# 1D integration
function integrate_1d(f, a, b, N, rule)
    if rule == :simpson13_close
        return simpson13_rule(f, a, b, N)

    elseif rule == :simpson13_open
        return simpson13_rule_min_open_max_open(f, a, b, N)

    elseif rule == :simpson38_close
        return simpson38_rule(f, a, b, N)

    elseif rule == :simpson38_open
        return simpson38_rule_min_open_max_open(f, a, b, N)

    elseif rule == :bode_close
        return bode_rule(f, a, b, N)

    elseif rule == :bode_open
        return bode_rule_min_open_max_open(f, a, b, N)

    else
        error("Unknown integration rule: $rule")
    end
end

# ============================================================
# 2D tensor-product quadrature
# ============================================================
function integrate_2d(f, a, b, N, rule)

    xs, wx = quadrature_1d_nodes_weights(a, b, N, rule)
    ys, wy = xs, wx   # same bounds

    total = 0.0

    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            total += wi * wy[j] * f(xi, ys[j])
        end
    end

    return total
end

# ============================================================
# 3D tensor-product quadrature
# ============================================================
function integrate_3d(f, a, b, N, rule)

    xs, wx = quadrature_1d_nodes_weights(a, b, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx

    total = 0.0

    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            yj = ys[j]
            wij = wi * wy[j]
            for k in eachindex(zs)
                total += wij * wz[k] * f(xi, yj, zs[k])
            end
        end
    end

    return total
end

# ============================================================
# 4D tensor-product quadrature
# ============================================================
function integrate_4d(f, a, b, N, rule)

    xs, wx = quadrature_1d_nodes_weights(a, b, N, rule)
    ys, wy = xs, wx
    zs, wz = xs, wx
    ts, wt = xs, wx

    total = 0.0

    @inbounds for i in eachindex(xs)
        xi = xs[i]
        wi = wx[i]
        for j in eachindex(ys)
            yj = ys[j]
            wij = wi * wy[j]
            for k in eachindex(zs)
                zk = zs[k]
                wijk = wij * wz[k]
                for l in eachindex(ts)
                    total += wijk * wt[l] * f(xi, yj, zk, ts[l])
                end
            end
        end
    end

    return total
end

# ============================================================
# Tensor-product foundation: generate nodes & weights
# (Supports BOTH close and open rules)
# ============================================================
function quadrature_1d_nodes_weights(a::Real, b::Real, N::Int, rule::Symbol)

    aa = float(a)
    bb = float(b)
    h  = (bb - aa)/N

    xs = Float64[]
    ws = Float64[]

    # -----------------------------
    # Simpson 1/3 Close
    # -----------------------------
    if rule == :simpson13_close
        N % 2 == 0 || error("Simpson 1/3 requires N divisible by 2")

        for j in 0:N
            push!(xs, aa + j*h)
            if j==0 || j==N
                push!(ws, 1.0)
            elseif isodd(j)
                push!(ws, 4.0)
            else
                push!(ws, 2.0)
            end
        end
        ws .*= (h/3)

    # -----------------------------
    # Simpson 3/8 Close
    # -----------------------------
    elseif rule == :simpson38_close
        N % 3 == 0 || error("Simpson 3/8 requires N divisible by 3")

        for j in 0:N
            push!(xs, aa + j*h)
            if j==0 || j==N
                push!(ws, 1.0)
            elseif j%3==0
                push!(ws, 2.0)
            else
                push!(ws, 3.0)
            end
        end
        ws .*= (3h/8)

    # -----------------------------
    # Bode Close
    # -----------------------------
    elseif rule == :bode_close
        N % 4 == 0 || error("Bode requires N divisible by 4")

        for j in 0:N
            push!(xs, aa + j*h)
            if j==0 || j==N
                push!(ws, 7.0)
            elseif isodd(j)
                push!(ws, 32.0)
            elseif j%4==2
                push!(ws, 12.0)
            else
                push!(ws, 14.0)
            end
        end
        ws .*= (2h/45)

    # -----------------------------
    # Simpson 1/3 Open-chain
    # (min-open max-open)
    # nodes: interior only
    # -----------------------------
    elseif rule == :simpson13_open
        N % 2 == 0 || error("Simpson13_open requires even N")

        for j in 1:N-1
            push!(xs, aa + j*h)
            if isodd(j)
                push!(ws, 4.0)
            else
                push!(ws, 2.0)
            end
        end
        ws .*= (h/3)

    # -----------------------------
    # Simpson 3/8 Open-chain
    # -----------------------------
    elseif rule == :simpson38_open
        N % 4 == 0 || error("Simpson38_open requires N divisible by 4")

        for j in 1:N-1
            push!(xs, aa + j*h)
            if j%4==2
                push!(ws, 2.0)
            else
                push!(ws, 3.0)
            end
        end
        ws .*= (3h/8)

    # -----------------------------
    # Bode Open-chain
    # -----------------------------
    elseif rule == :bode_open
        N % 4 == 0 || error("Bode_open requires N divisible by 4")

        for j in 1:N-1
            push!(xs, aa + j*h)
            if j%4==0
                push!(ws, 14.0)
            elseif j%2==0
                push!(ws, 12.0)
            else
                push!(ws, 32.0)
            end
        end
        ws .*= (2h/45)

    else
        error("Unknown rule $rule")
    end

    return xs, ws
end

end