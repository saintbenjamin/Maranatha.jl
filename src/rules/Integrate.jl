# ============================================================================
# src/rules/Integrate.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Integrate

using ..Simpson13Rule, ..Simpson38Rule, ..BodeRule
using ..Simpson13Rule_MinOpen_MaxOpen, ..Simpson38Rule_MinOpen_MaxOpen, ..BodeRule_MinOpen_MaxOpen

export integrate_nd, quadrature_1d_nodes_weights

"""
    integrate_nd(integrand, a, b, N, dim, rule)

Dispatch an integration routine by dimension and quadrature rule.

# Function description
This function selects a tensor-product quadrature implementation for `dim = 2, 3, 4`,
and a direct 1D rule implementation for `dim = 1`. The integration bounds `[a, b]`
are applied identically along every axis, i.e. the domain is `[a, b]^dim`.

# Arguments
- `integrand`: Integrand function of `dim` variables:
  - `dim == 1` → `integrand(x)`
  - `dim == 2` → `integrand(x, y)`
  - `dim == 3` → `integrand(x, y, z)`
  - `dim == 4` → `integrand(x, y, z, t)`
- `a`, `b`: Integration bounds (applied to every dimension).
- `N`: Number of intervals per axis (must satisfy rule-specific constraints).
- `dim`: Dimensionality (`1`, `2`, `3`, or `4`).
- `rule`: Integration rule symbol (e.g. `:simpson13_close`, `:simpson38_close`, `:bode_close`,
  and the corresponding `*_open` variants).

# Returns
- Estimated integral value as a `Float64`.

# Errors
- Throws an error if `dim` is not in `{1,2,3,4}`.
- Throws an error if `rule` is not supported by the underlying implementation
  for the selected dimension.
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

"""
    integrate_1d(f, a, b, N, rule) -> Float64

Evaluate a 1D integral of `f(x)` over `[a, b]` using the specified quadrature rule.

# Function description
This function dispatches to the dedicated 1D implementations for each supported
Newton–Cotes rule. Both closed and open-chain variants are supported.

# Arguments
- `f`: Integrand function `f(x)`.
- `a`, `b`: Integration bounds.
- `N`: Number of intervals (rule-specific divisibility/minimum constraints apply).
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.

# Errors
- Throws an error if `rule` is not recognized.
"""
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

"""
    integrate_2d(f, a, b, N, rule) -> Float64

Evaluate a 2D integral of `f(x, y)` over the square domain `[a, b] × [a, b]`
using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using
`quadrature_1d_nodes_weights(a, b, N, rule)` and forms the tensor product:
`Σ_i Σ_j w_i w_j f(x_i, y_j)`.

Loop ordering and accumulation are preserved exactly as implemented.

# Arguments
- `f`: 2D integrand function `f(x, y)`.
- `a`, `b`: Square domain bounds (used for both axes).
- `N`: Number of intervals per axis.
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.
"""
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

"""
    integrate_3d(f, a, b, N, rule) -> Float64

Evaluate a 3D integral of `f(x, y, z)` over the cube domain `[a, b]^3`
using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using
`quadrature_1d_nodes_weights(a, b, N, rule)` and forms the tensor product:
`Σ_i Σ_j Σ_k w_i w_j w_k f(x_i, y_j, z_k)`.

Loop ordering and accumulation are preserved exactly as implemented.

# Arguments
- `f`: 3D integrand function `f(x, y, z)`.
- `a`, `b`: Cube domain bounds (used for all axes).
- `N`: Number of intervals per axis.
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.
"""
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

"""
    integrate_4d(f, a, b, N, rule) -> Float64

Evaluate a 4D integral of `f(x, y, z, t)` over the hypercube domain `[a, b]^4`
using a tensor-product quadrature constructed from 1D nodes and weights.

# Function description
This routine generates 1D quadrature nodes and weights using
`quadrature_1d_nodes_weights(a, b, N, rule)` and forms the tensor product:
`Σ_i Σ_j Σ_k Σ_l w_i w_j w_k w_l f(x_i, y_j, z_k, t_l)`.

Loop ordering and accumulation are preserved exactly as implemented.

# Arguments
- `f`: 4D integrand function `f(x, y, z, t)`.
- `a`, `b`: Hypercube domain bounds (used for all axes).
- `N`: Number of intervals per axis.
- `rule`: Integration rule symbol.

# Returns
- Estimated integral value as a `Float64`.
"""
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

"""
    quadrature_1d_nodes_weights(a::Real, b::Real, N::Int, rule::Symbol) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate 1D quadrature nodes and weights for a Newton–Cotes rule on `[a, b]`.

# Function description
This function constructs the node vector `xs` and weight vector `ws` for the
specified `rule` over the interval `[a, b]` using a uniform step size
`h = (b-a)/N`. The returned `(xs, ws)` pair is suitable for building
tensor-product quadrature rules in higher dimensions.

Both closed rules (including endpoints) and the custom open-chain rules
(excluding endpoints) are supported. The implementation preserves the exact
node ordering and weight construction used by the original code.

# Arguments
- `a`, `b`: Integration bounds.
- `N`: Number of intervals defining the uniform grid spacing `h = (b-a)/N`.
- `rule`: Quadrature rule symbol:
  - Closed rules: `:simpson13_close`, `:simpson38_close`, `:bode_close`
  - Open-chain rules: `:simpson13_open`, `:simpson38_open`, `:bode_open`

# Returns
- `xs::Vector{Float64}`: Quadrature nodes (in the exact order they are pushed).
- `ws::Vector{Float64}`: Corresponding quadrature weights (already scaled by `h`
  and any rule-specific prefactors).

# Errors
- Throws an error if `rule` is not recognized.
- Throws an error if `N` violates rule-specific constraints (divisibility and/or minimum `N`).
"""
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
    # -----------------------------
    elseif rule == :simpson13_open
        (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
        (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")

        # Nodes used:
        #   j = 1, 3, even 4..N-4, odd 5..N-5, N-3, N-1
        # Coefficients (inside bracket) multiplied by h:
        #   (9/4) at j=1 and j=N-1
        #   (13/12) at j=3 and j=N-3
        #   (4/3) for even j=4,6,...,N-4
        #   (2/3) for odd  j=5,7,...,N-5

        push!(xs, aa + 1*h);          push!(ws, 9.0/4.0)
        push!(xs, aa + 3*h);          push!(ws, 13.0/12.0)

        @inbounds for j in 4:2:(N-4)
            push!(xs, aa + j*h);      push!(ws, 4.0/3.0)
        end

        @inbounds for j in 5:2:(N-5)
            push!(xs, aa + j*h);      push!(ws, 2.0/3.0)
        end

        push!(xs, aa + (N-3)*h);      push!(ws, 13.0/12.0)
        push!(xs, aa + (N-1)*h);      push!(ws, 9.0/4.0)

        ws .*= h

    # -----------------------------
    # Simpson 3/8 Open-chain
    # -----------------------------
    elseif rule == :simpson38_open
        (N % 4 == 0) || error("Open 3-point chained rule requires N divisible by 4, got N = $N")
        (N >= 4)     || error("Open 3-point chained rule requires N ≥ 4, got N = $N")

        # Panels of width 4h:
        # per panel k: j1=4k+1, j2=4k+2, j3=4k+3
        # coefficient times h:
        #   (8/3) f(x_{j1}) + (-4/3) f(x_{j2}) + (8/3) f(x_{j3})

        M = N ÷ 4
        for k in 0:(M-1)
            j1 = 4k + 1
            j2 = 4k + 2
            j3 = 4k + 3

            push!(xs, aa + j1*h);     push!(ws,  8.0/3.0)
            push!(xs, aa + j2*h);     push!(ws, -4.0/3.0)
            push!(xs, aa + j3*h);     push!(ws,  8.0/3.0)
        end

        ws .*= h

    # -----------------------------
    # Bode Open-chain
    # -----------------------------
    elseif rule == :bode_open
        (N % 4 == 0) || error("Open composite Boole requires N divisible by 4, got N = $N")
        (N >= 16)    || error("Open composite Boole requires N ≥ 16 (non-overlapping end stencils), got N = $N")

        # Closed interior weights by j mod 4 (endpoints excluded)
        w_mod13 = 64.0 / 45.0
        w_mod2  =  8.0 / 15.0
        w_mod0  = 28.0 / 45.0

        w_end = 14.0 / 45.0
        c1, c2, c3, c4, c5, c6 = 6.0, -15.0, 20.0, -15.0, 6.0, -1.0

        @inline function w_closed(j::Int)::Float64
            r = j % 4
            if r == 0
                return w_mod0
            elseif r == 2
                return w_mod2
            else
                return w_mod13
            end
        end

        # Modified left-end weights for j=1..6: w'_j = w_closed(j) + w_end*cj
        wL = Vector{Float64}(undef, 6)
        wL[1] = w_closed(1) + w_end*c1
        wL[2] = w_closed(2) + w_end*c2
        wL[3] = w_closed(3) + w_end*c3
        wL[4] = w_closed(4) + w_end*c4
        wL[5] = w_closed(5) + w_end*c5
        wL[6] = w_closed(6) + w_end*c6

        # Left stencil: j=1..6
        @inbounds for j in 1:6
            push!(xs, aa + j*h)
            push!(ws, wL[j])
        end

        # Middle: j=7..N-7
        @inbounds for j in 7:(N-7)
            push!(xs, aa + j*h)
            push!(ws, w_closed(j))
        end

        # Right stencil: exactly your loop k=6:-1:1, j=N-k, weight=wL[k]
        @inbounds for k in 6:-1:1
            j = N - k
            push!(xs, aa + j*h)
            push!(ws, wL[k])
        end

        ws .*= h

    else
        error("Unknown rule $rule")
    end

    return xs, ws
end

end  # module Integrate