# ============================================================================
# src/Quadrature/Gauss.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Gauss

using ..LinearAlgebra

# ============================================================================
# Float64-only composite Gaussian families with boundary patterns:
#   LORO -> Gauss-Legendre
#   LCRO -> Gauss-Radau (left endpoint included)
#   LORC -> Gauss-Radau (right endpoint included)
#   LCRC -> Gauss-Lobatto (both endpoints included)
# ============================================================================

# ------------------------------------------------------------
# Legendre helpers (standard P_n, NOT normalized)
# ------------------------------------------------------------

"""
    _legendre_Pn_Pn1(n::Int, x::Float64) -> (P_n, P_{n-1})

Evaluate the *standard* (non-normalized) Legendre polynomial `P_n(x)` and the
previous order `P_{n-1}(x)` at a scalar point `x ∈ ℝ`.

# Function description
This routine uses the classic three-term recurrence:
```math
P_0(x)=1,\\quad P_1(x)=x,\\quad
P_k(x)=\\frac{(2k-1)xP_{k-1}(x)-(k-1)P_{k-2}(x)}{k}.
```

and returns the pair `(P_n(x), P_{n-1}(x))`.

This is a low-level helper used by Newton iterations in Radau/Lobatto root solves.

# Arguments

* `n`: Order (must satisfy `n ≥ 0`).
* `x`: Evaluation point.

# Returns

* `Tuple{Float64,Float64}`: `(P_n(x), P_{n-1}(x))`.

# Errors

* Throws `error("n must be ≥ 0")` if `n < 0`.

# Notes

* This is **not** the orthonormal Legendre basis; it is the conventional `P_n`.
"""
@inline function _legendre_Pn_Pn1(n::Int, x::Float64)::Tuple{Float64,Float64}
    # returns (P_n(x), P_{n-1}(x))
    n >= 0 || error("n must be ≥ 0")
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end

    Pnm1 = 1.0    # P0
    Pn   = x      # P1
    for k in 2:n
        Pkp = ((2k - 1) * x * Pn - (k - 1) * Pnm1) / k
        Pnm1 = Pn
        Pn   = Pkp
    end
    return Pn, Pnm1
end

"""
    _legendre_Pn_deriv(n::Int, x::Float64) -> (P_n, P_n′)

Evaluate the standard Legendre polynomial `P_n(x)` and its derivative `P_n′(x)`.

# Function description

This routine computes `P_n(x)` (and `P_{n-1}(x)`) via [`_legendre_Pn_Pn1`](@ref),
then uses the identity:

```math
P_n'(x) = \\frac{n}{x^2 - 1}\\left(xP_n(x) - P_{n-1}(x)\\right).
```

This formula is numerically sensitive very near `x = ±1` due to division by `(x^2-1)`.
Callers should avoid evaluating exactly at endpoints during Newton steps (see
[`_clamp_open`](@ref)).

# Arguments

* `n`: Order (must satisfy `n ≥ 0`).
* `x`: Evaluation point.

# Returns

* `Tuple{Float64,Float64}`: `(P_n(x), P_n'(x))`.

# Notes

* For `n == 0`, returns `(1.0, 0.0)`.
"""
@inline function _legendre_Pn_deriv(n::Int, x::Float64)::Tuple{Float64,Float64}
    # returns (P_n(x), P_n'(x))
    (Pn, Pn1) = _legendre_Pn_Pn1(n, x)
    if n == 0
        return Pn, 0.0
    end
    # Pn'(x) = n/(x^2 - 1) * (x*Pn - P_{n-1})
    den = x*x - 1.0
    # avoid blow-up extremely near ±1 in Newton; caller clamps away from endpoints
    dPn = n * (x * Pn - Pn1) / den
    return Pn, dPn
end

"""
    _clamp_open(x::Float64) -> Float64

Clamp a Newton iterate away from the singular endpoints `x = ±1`.

# Function description

Several derivative identities used in Gauss-family root finding contain the factor
`1/(x^2 - 1)`, which becomes singular at `x = ±1`. During Newton iterations, it is
common to step extremely close to endpoints due to roundoff.

This helper clamps `x` into the open interval:

```math
(-1 + \\varepsilon,\\; 1 - \\varepsilon)
```

where `ε = 64*eps(Float64)`.

# Arguments

* `x`: Newton iterate.

# Returns

* `Float64`: Clamped value in the safe open interval.

# Notes

* This is purely numerical hygiene; it does not change the mathematical definition
  of the quadrature family.
"""
@inline function _clamp_open(x::Float64)::Float64
    # keep Newton away from ±1 where derivative formula divides by (x^2-1)
    # (this is purely numerical hygiene)
    tiny = 64.0 * eps(Float64)
    x <= -1.0 + tiny && return -1.0 + tiny
    x >=  1.0 - tiny && return  1.0 - tiny
    return x
end

# ------------------------------------------------------------
# 1) Gauss–Legendre (Golub–Welsch) on [-1,1]
# ------------------------------------------------------------

"""
    gauss_legendre_nodes_weights_float(n::Int) -> (nodes, weights)

Compute `n`-point Gauss–Legendre quadrature nodes and weights on `[-1, 1]`
in `Float64`.

# Function description

This implementation uses the Golub–Welsch algorithm (eigen-decomposition of a
symmetric tridiagonal Jacobi matrix).

For Legendre weight `w(x)=1` on `[-1,1]`, the orthonormal recurrence has:

* `a_k = 0`
* `b_k = k / sqrt(4k^2 - 1)` for `k = 1..n-1`

Construct:

```math
J = \\mathrm{SymTridiagonal}(a, b),
```

then:

* nodes `t_i` are eigenvalues of `J`,
* weights `w_i = μ_0 (v_{1,i})^2` where `μ_0 = ∫_{-1}^1 1 dx = 2`
  and `v_{1,i}` is the first component of the `i`-th normalized eigenvector.

# Arguments

* `n`: Number of quadrature points (must satisfy `n ≥ 1`).

# Returns

* `nodes::Vector{Float64}`: Length `n`, sorted ascending.
* `weights::Vector{Float64}`: Length `n`, aligned with `nodes`.

# Errors

* Throws `error("n must be ≥ 1")` if `n < 1`.

# Notes

* This returns the *single-interval* Gauss–Legendre rule on `[-1,1]`.
  Composite usage is handled by `_composite_gauss_nodes_weights` / `_composite_gauss_u_grid`.
"""
function gauss_legendre_nodes_weights_float(n::Int)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 1 || error("n must be ≥ 1")

    # For Legendre w(x)=1 on [-1,1], orthonormal recurrence has:
    #   a_k = 0
    #   b_k = k / sqrt(4k^2 - 1),  k=1..n-1
    #
    # J = SymTridiagonal(a, b)
    # nodes = eigenvalues(J)
    # weights = μ0 * (v1_i)^2, μ0 = ∫_{-1}^{1} 1 dx = 2
    a = zeros(Float64, n)
    b = Vector{Float64}(undef, n-1)
    @inbounds for k in 1:(n-1)
        b[k] = k / sqrt(4.0*k*k - 1.0)
    end

    J = SymTridiagonal(a, b)
    E = eigen(J)  # sorted ascending
    t = Vector{Float64}(E.values)
    V = E.vectors
    w = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        v1 = V[1, i]
        w[i] = 2.0 * (v1 * v1)
    end
    return t, w
end

# ------------------------------------------------------------
# 2) Gauss–Radau (one endpoint fixed)
#
# Facts (Legendre weight 1):
# - Left Radau (includes x=-1): remaining roots are zeros of  F(x)=P_n(x)+P_{n-1}(x)
# - Right Radau(includes x=+1): remaining roots are zeros of F(x)=P_n(x)-P_{n-1}(x)
# - Endpoint weight: 2/n^2
# - Other weights:
#     left  (-1 fixed): w_i = (1 + x_i) / (n^2 * [P_{n-1}(x_i)]^2)
#     right (+1 fixed): w_i = (1 - x_i) / (n^2 * [P_{n-1}(x_i)]^2)
# ------------------------------------------------------------

"""
    gauss_radau_left_nodes_weights_float(n::Int) -> (nodes, weights)

Compute `n`-point Gauss–Radau quadrature on `[-1,1]` that **includes the left endpoint** `x=-1`
in `Float64`.

# Function description

For Legendre weight `w(x)=1`:

* One node is fixed at `x=-1`.
* The remaining `(n-1)` interior nodes are roots of:

```math
F(x) = P_n(x) + P_{n-1}(x).
```

This routine finds those roots by Newton iteration, using Gauss–Legendre `(n-1)` nodes
as initial guesses (a practical robust choice).

Weights:

* Endpoint weight:

```math
w_1 = \\frac{2}{n^2}
```

* Interior weights:

```math
w_i = \\frac{1 + x_i}{n^2 [P_{n-1}(x_i)]^2}.
```

# Arguments

* `n`: Number of quadrature points (must satisfy `n ≥ 2`).

# Returns

* `nodes::Vector{Float64}`: Length `n`, sorted ascending, includes `-1.0`.
* `weights::Vector{Float64}`: Length `n`, aligned with `nodes`.

# Errors

* Throws `error("Radau needs n ≥ 2 ...")` if `n < 2`.

# Numerical notes

* Newton steps are clamped away from `±1` via [`_clamp_open`](@ref) to avoid
  derivative singularities.
* The iteration budget is fixed (80 steps); it breaks early when `|dx|` is small.
"""
function gauss_radau_left_nodes_weights_float(n::Int)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Radau needs n ≥ 2 (got n=$n)")

    # nodes: x1=-1 plus (n-1) roots of P_n + P_{n-1}
    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[1] = -1.0
    w[1] = 2.0 / (Float64(n) * Float64(n))

    # initial guesses: use GL(n-1) nodes as seeds (works well in practice)
    seeds, _ = gauss_legendre_nodes_weights_float(n-1)

    # Newton for each root in (-1,1)
    @inbounds for i in 1:(n-1)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pn, dPn)     = _legendre_Pn_deriv(n, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            F  = Pn + Pnm1
            dF = dPn + dPnm1

            dx = F / dF
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end
        t[i+1] = x

        # weight for interior node
        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i+1] = (1.0 + x) / ((Float64(n)^2) * (Pnm1*Pnm1))
    end

    # sort by nodes (keep weights aligned)
    p = sortperm(t)
    return t[p], w[p]
end

"""
    gauss_radau_right_nodes_weights_float(n::Int) -> (nodes, weights)

Compute `n`-point Gauss–Radau quadrature on `[-1,1]` that **includes the right endpoint** `x=+1`
in `Float64`.

# Function description

For Legendre weight `w(x)=1`:

* One node is fixed at `x=+1`.
* The remaining `(n-1)` interior nodes are roots of:

```math
F(x) = P_n(x) - P_{n-1}(x).
```

This routine finds those roots by Newton iteration, using Gauss–Legendre `(n-1)` nodes
as initial guesses.

Weights:

* Endpoint weight:

```math
w_n = \\frac{2}{n^2}
```

* Interior weights:

```math
w_i = \\frac{1 - x_i}{n^2 [P_{n-1}(x_i)]^2}.
```

# Arguments

* `n`: Number of quadrature points (must satisfy `n ≥ 2`).

# Returns

* `nodes::Vector{Float64}`: Length `n`, sorted ascending, includes `+1.0`.
* `weights::Vector{Float64}`: Length `n`, aligned with `nodes`.

# Errors

* Throws `error("Radau needs n ≥ 2 ...")` if `n < 2`.

# Numerical notes

* Newton steps are clamped away from `±1` via [`_clamp_open`](@ref).
"""
function gauss_radau_right_nodes_weights_float(n::Int)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Radau needs n ≥ 2 (got n=$n)")

    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[end] = 1.0
    w[end] = 2.0 / (Float64(n) * Float64(n))

    seeds, _ = gauss_legendre_nodes_weights_float(n-1)

    @inbounds for i in 1:(n-1)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pn, dPn)     = _legendre_Pn_deriv(n, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            F  = Pn - Pnm1
            dF = dPn - dPnm1

            dx = F / dF
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end
        t[i] = x

        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i] = (1.0 - x) / ((Float64(n)^2) * (Pnm1*Pnm1))
    end

    p = sortperm(t)
    return t[p], w[p]
end

# ------------------------------------------------------------
# 3) Gauss–Lobatto (both endpoints fixed)
#
# Facts:
# - n nodes includes both endpoints: x1=-1, xn=+1
# - interior nodes are roots of P'_{n-1}(x)=0
#   Use equivalent G(x) = P_{n-2}(x) - x*P_{n-1}(x) = 0 for interior roots
#   because (1-x^2)P'_{n-1} = (n-1)(P_{n-2} - x P_{n-1})
# - weights:
#     w1 = wn = 2/(n(n-1))
#     interior: w_i = 2 / (n(n-1) * [P_{n-1}(x_i)]^2)
# ------------------------------------------------------------

"""
    gauss_lobatto_nodes_weights_float(n::Int) -> (nodes, weights)

Compute `n`-point Gauss–Lobatto quadrature on `[-1,1]` that **includes both endpoints**
`x=-1` and `x=+1`, in `Float64`.

# Function description

For Legendre weight `w(x)=1`:

* Endpoints are fixed nodes: `x_1 = -1`, `x_n = +1`.
* The interior `(n-2)` nodes are roots of `P'_{n-1}(x)=0`.
  This implementation instead solves the equivalent equation:

```math
G(x) = P_{n-2}(x) - x P_{n-1}(x) = 0,
```

since:

```math
(1-x^2)P'_{n-1}(x) = (n-1)\\bigl(P_{n-2}(x) - xP_{n-1}(x)\\bigr).
```

Roots are found by Newton iteration with Gauss–Legendre `(n-2)` nodes as seeds.

Weights:

* Endpoint weights:

```math
w_1 = w_n = \\frac{2}{n(n-1)}.
```

* Interior weights:

```math
w_i = \\frac{2}{n(n-1)[P_{n-1}(x_i)]^2}.
```

# Arguments

* `n`: Number of quadrature points (must satisfy `n ≥ 2`).

# Returns

* `nodes::Vector{Float64}`: Length `n`, sorted ascending, includes `±1.0`.
* `weights::Vector{Float64}`: Length `n`, aligned with `nodes`.

# Errors

* Throws `error("Lobatto needs n ≥ 2 ...")` if `n < 2`.

# Special cases

* If `n == 2`, returns only endpoints with equal weights.
"""
function gauss_lobatto_nodes_weights_float(n::Int)::Tuple{Vector{Float64},Vector{Float64}}
    n >= 2 || error("Lobatto needs n ≥ 2 (got n=$n)")

    t = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    t[1]   = -1.0
    t[end] =  1.0

    nn = Float64(n)
    w[1]   = 2.0 / (nn * (nn - 1.0))
    w[end] = w[1]

    if n == 2
        # just endpoints
        return t, w
    end

    # seeds: GL(n-2) as starting points for interior roots
    seeds, _ = gauss_legendre_nodes_weights_float(n-2)

    # Newton on G(x) = P_{n-2}(x) - x P_{n-1}(x)
    @inbounds for i in 1:(n-2)
        x = _clamp_open(seeds[i])
        for _ in 1:80
            (Pnm2, dPnm2) = _legendre_Pn_deriv(n-2, x)
            (Pnm1, dPnm1) = _legendre_Pn_deriv(n-1, x)

            G  = Pnm2 - x * Pnm1
            dG = dPnm2 - (Pnm1 + x * dPnm1)

            dx = G / dG
            x -= dx
            x = _clamp_open(x)
            abs(dx) < 1e4 * eps(Float64) && break
        end

        t[i+1] = x

        Pnm1, _ = _legendre_Pn_deriv(n-1, x)
        w[i+1] = 2.0 / (nn * (nn - 1.0) * (Pnm1*Pnm1))
    end

    p = sortperm(t)
    return t[p], w[p]
end

"""
    _GAUSS_CACHE :: Dict{Tuple{Int,Symbol}, (nodes, weights)}

Process-local cache for single-interval Gauss-family nodes/weights on `[-1,1]`.

# Description

This cache stores the output of [`_gauss_family_nodes_weights`](@ref) for a given pair:

* `n`       : number of points in the Gauss family rule,
* `boundary`: which family (`:LORO`, `:LCRO`, `:LORC`, `:LCRC`).

The stored value is:

* `(nodes::Vector{Float64}, weights::Vector{Float64})` on `[-1,1]`.

# Purpose

Computing nodes/weights can be moderately expensive (eigen-decomposition, Newton solves).
Caching avoids repeated recomputation when composite quadrature repeatedly requests the same
family configuration.

# Notes

* This cache is not persistent across Julia sessions.
* Keys are intentionally small: `(n, boundary)`.
"""
const _GAUSS_CACHE = Dict{Tuple{Int,Symbol}, Tuple{Vector{Float64},Vector{Float64}}}()

"""
    _is_gauss_rule(rule::Symbol) -> Bool

Return `true` if `rule` is a Gauss-family rule symbol of the form `:gauss_pK`.

# Function description

This module encodes "number of Gauss points per block" into the symbol string:

* `:gauss_p2`, `:gauss_p3`, ...

This helper checks whether the symbol string begins with `"gauss_p"`.

# Arguments

* `rule`: Rule symbol.

# Returns

* `Bool`: `true` if `rule` starts with `"gauss_p"`, else `false`.

# Notes

* This only validates the *prefix*; parsing the integer is handled by
  [`_parse_gauss_p`](@ref).
"""
@inline function _is_gauss_rule(rule::Symbol)::Bool
    startswith(String(rule), "gauss_p")
end

"""
    _parse_gauss_p(rule::Symbol) -> Int

Parse the number of points `n` from a Gauss-family rule symbol `:gauss_pN`.

# Function description

Expected encoding:

* `:gauss_p1`, `:gauss_p2`, `:gauss_p3`, ...

This function:

1. verifies the prefix via [`_is_gauss_rule`](@ref),
2. parses the substring after `"gauss_p"` as an integer `n`,
3. validates `n ≥ 1`.

# Arguments

* `rule`: Rule symbol.

# Returns

* `Int`: Parsed point count `n`.

# Errors

* Throws (via `JobLoggerTools.error_benji`) if prefix is wrong or `n` is invalid.

# Notes

* This helper is typically used by a higher-level quadrature dispatcher.
"""
function _parse_gauss_p(rule::Symbol)::Int
    _is_gauss_rule(rule) || JobLoggerTools.error_benji("rule must start with :gauss_p (got $rule)")
    s = String(rule)
    n = parse(Int, s[8:end])  # after "gauss_p"
    n >= 1 || JobLoggerTools.error_benji("gauss points must be ≥ 1 (got $n)")
    return n
end

"""
    _gauss_family_nodes_weights(n::Int, boundary::Symbol) -> (nodes, weights)

Return the single-interval Gauss-family nodes/weights on `[-1,1]`
for a specified boundary pattern.

# Function description

This routine chooses among the supported Gauss families:

* `boundary == :LORO` : Gauss–Legendre (no endpoints included)
* `boundary == :LCRO` : Gauss–Radau (left endpoint `-1` included)
* `boundary == :LORC` : Gauss–Radau (right endpoint `+1` included)
* `boundary == :LCRC` : Gauss–Lobatto (both endpoints included)

It uses and populates the cache [`_GAUSS_CACHE`](@ref) keyed by `(n, boundary)`.

# Arguments

* `n`: Points per rule (Legendre: `n ≥ 1`, Radau/Lobatto: `n ≥ 2`).
* `boundary`: One of `:LORO`, `:LCRO`, `:LORC`, `:LCRC`.

# Returns

* `nodes::Vector{Float64}`: Nodes on `[-1,1]` (sorted).
* `weights::Vector{Float64}`: Weights on `[-1,1]` aligned with nodes.

# Errors

* Throws (via `JobLoggerTools.error_benji`) if `boundary` is invalid or if `n` is too small
  for the requested family.
"""
function _gauss_family_nodes_weights(
    n::Int,
    boundary::Symbol
)::Tuple{Vector{Float64},Vector{Float64}}

    key = (n, boundary)
    cached = get(_GAUSS_CACHE, key, nothing)
    cached !== nothing && return cached

    t, w = if boundary === :LORO
        gauss_legendre_nodes_weights_float(n)
    elseif boundary === :LCRO
        n >= 2 || JobLoggerTools.error_benji("Gauss-Radau requires n ≥ 2 (got n=$n)")
        gauss_radau_left_nodes_weights_float(n)
    elseif boundary === :LORC
        n >= 2 || JobLoggerTools.error_benji("Gauss-Radau requires n ≥ 2 (got n=$n)")
        gauss_radau_right_nodes_weights_float(n)
    elseif boundary === :LCRC
        n >= 2 || JobLoggerTools.error_benji("Gauss-Lobatto requires n ≥ 2 (got n=$n)")
        gauss_lobatto_nodes_weights_float(n)
    else
        JobLoggerTools.error_benji("boundary must be :LCRC|:LORC|:LCRO|:LORO (got $boundary)")
    end

    _GAUSS_CACHE[key] = (t, w)
    return t, w
end

"""
    _composite_gauss_nodes_weights(a, b, N, npts, boundary) -> (xs, ws)

Construct **composite** Gauss-family nodes and weights on `[a,b]` by repeating an `npts`-point
rule on each of `N` uniform subintervals.

# Function description

This routine:

1. converts `(a,b)` to `Float64`,
2. defines uniform subinterval width `h = (b-a)/N`,
3. obtains the reference nodes/weights `(t,w)` on `[-1,1]` for the chosen family
   via [`_gauss_family_nodes_weights`](@ref),
4. maps the rule to each block `[xL, xR]` with the affine transform:

```math
x = \\mathrm{mid} + \\mathrm{scale}\\,t,\\quad
dx = \\mathrm{scale}\\,dt,\\quad
\\mathrm{mid}=\\frac{x_L+x_R}{2},\\quad \\mathrm{scale}=\\frac{x_R-x_L}{2}=\\frac{h}{2}.
```

The output arrays have length `npts*N`, containing *all* per-block nodes and weights.

# Arguments

* `a`, `b`: Global integration interval endpoints (converted to `Float64`).
* `N`: Number of uniform blocks (`N ≥ 1` expected by caller).
* `npts`: Number of Gauss points per block (family-dependent constraints apply).
* `boundary`: Family selector (`:LORO`, `:LCRO`, `:LORC`, `:LCRC`).

# Returns

* `xs::Vector{Float64}`: Length `npts*N`, composite nodes on `[a,b]`.
* `ws::Vector{Float64}`: Length `npts*N`, composite weights.

# Notes

* This is “composite Gauss” in the sense of *repeating a fixed high-order rule* on each block,
  not “Gaussian quadrature on the whole interval at once”.
"""
function _composite_gauss_nodes_weights(
    a::Real,
    b::Real,
    N::Int,
    npts::Int,
    boundary::Symbol
)::Tuple{Vector{Float64},Vector{Float64}}

    aa = Float64(a)
    bb = Float64(b)
    h  = (bb - aa) / Float64(N)

    t, w = _gauss_family_nodes_weights(npts, boundary)

    xs = Vector{Float64}(undef, npts * N)
    ws = Vector{Float64}(undef, npts * N)

    half = 0.5
    idx = 1
    for blk in 0:(N-1)
        xL = aa + Float64(blk) * h
        xR = xL + h
        mid   = (xL + xR) * half
        scale = (xR - xL) * half  # = h/2

        @inbounds for i in 1:npts
            xs[idx] = mid + scale * t[i]
            ws[idx] = scale * w[i]
            idx += 1
        end
    end

    return xs, ws
end

# ---- composite tiling on dimensionless u ∈ [0, N] with N unit blocks ----
# Map each block [m, m+1]:
#   u = m + (t+1)/2,   du = (1/2) dt

"""
    _composite_gauss_u_grid(N, npts, boundary) -> (U, W)

Build a *dimensionless* composite Gauss grid on `u ∈ [0, N]` by repeating an `npts`-point
rule on each unit block `[m, m+1]`.

# Function description

This is the dimensionless analogue of [`_composite_gauss_nodes_weights`](@ref), intended for
pipelines that factor the mapping `[a,b] ↔ u` separately.

For each block index `m = 0..N-1`, map reference rule `(t,w)` on `[-1,1]` to `[m, m+1]` via:

```math
u = m + \\frac{t+1}{2},\\quad du = \\frac{1}{2}dt.
```

Thus, weights are scaled by `1/2`.

# Arguments

* `N`: Number of unit blocks (must satisfy `N ≥ 1`).
* `npts`: Points per block (family-dependent constraints apply).
* `boundary`: Family selector (`:LORO`, `:LCRO`, `:LORC`, `:LCRC`).

# Returns

* `U::Vector{Float64}`: Length `npts*N`, nodes on `[0,N]`.
* `W::Vector{Float64}`: Length `npts*N`, weights such that `∫_0^N f(u)du ≈ Σ f(U_i) W_i`.

# Errors

* Throws `ArgumentError("N must be ≥ 1")` if `N < 1`.

# Notes

* This function does *not* include any global scaling by `(b-a)`; it is purely on the `u` grid.
"""
function _composite_gauss_u_grid(
    N::Int,
    npts::Int,
    boundary::Symbol
)::Tuple{Vector{Float64},Vector{Float64}}

    N >= 1 || throw(ArgumentError("N must be ≥ 1"))

    t, w = _gauss_family_nodes_weights(npts, boundary)  # on [-1,1]

    U = Vector{Float64}(undef, npts * N)
    W = Vector{Float64}(undef, npts * N)

    half = 0.5
    idx = 1
    for m in 0:(N-1)
        mm = Float64(m)
        @inbounds for i in 1:npts
            U[idx] = mm + (t[i] + 1.0) * half
            W[idx] = w[i] * half
            idx += 1
        end
    end

    return U, W
end

end  # module Gauss