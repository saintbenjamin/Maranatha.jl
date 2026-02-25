# ============================================================================
# src/error/ErrorEstimator/estimate_error_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    estimate_error_nd(
        f,
        a::Real,
        b::Real,
        N::Int,
        rule::Symbol;
        dim::Int
    ) -> Float64

Estimate an **N-dimensional integration error scale** over the hypercube
domain `[a,b]^dim` using a derivative-based tensor-product heuristic with
optional boundary-difference handling for selected open-chain rules.

# Function description
This routine generalizes the legacy dimension-specific estimators
(`estimate_error_1d/2d/3d/4d`) into a single dimension-agnostic model.

The returned value is a **heuristic signed error scale**, intended mainly
for stabilizing convergence fits and extrapolation models rather than
providing a strict truncation bound.

Two regimes are supported:

---

## (A) Boundary-difference model (selected open-chain rules)
For rules where `_has_boundary_error_model(rule)` is true, a boundary-based
leading-term model is applied **axis by axis**.

For each axis `μ = 1..dim`, define boundary points

`xL = a + off*h`,  
`xR = a + (N-off)*h`,  

where `h = (b-a)/N` and `(p, K, dord, off)` are returned by
`_boundary_error_params(rule)`.

The estimator constructs

```
I_μ = ∫_{[a,b]^{dim-1}}
[ ∂_μ^{dord} f(…, xL, …)
- ∂_μ^{dord} f(…, xR, …) ]
d(other axes)
```

and returns

`E ≈ K * h^p * Σ_μ I_μ`.

This boundary-difference structure models the dominant truncation behavior
of endpoint-free chained Newton–Cotes formulas and typically improves χ²
stability during extrapolation.

Rule-specific constraints on `N` (such as divisibility or minimum size)
are enforced explicitly in this branch.

---

## (B) Default midpoint tensor-style model (all other supported rules)
Otherwise, the estimator falls back to a midpoint tensor-product heuristic:

1. Build 1D quadrature nodes and weights `(xs, ws)` via
   `quadrature_1d_nodes_weights`.
2. For each axis `μ`, compute an axis-wise contribution

```
I_μ = ∫_{[a,b]^{dim-1}} ∂_μ^m f(x̄_μ) d(other axes),
```

where `x̄ = (a+b)/2` is the midpoint along the differentiated axis.

3. Return the scale model

`E ≈ C * (b-a) * h^m * Σ_μ I_μ`,

with `(m, C) = _rule_params_for_tensor_error(rule)`.

This reproduces the legacy midpoint tensor-style accumulation used in
lower-dimensional implementations.

---

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Domain bounds defining the hypercube `[a,b]^dim`.
- `N`: Number of subdivisions per axis (`h = (b-a)/N`).
- `rule`: Integration rule symbol.
- `dim`: Number of spatial dimensions (must satisfy `dim ≥ 1`).

# Returns
- `Float64`: Heuristic signed error estimate (scale model).
  If the rule has no tensor parameters (`m == 0`), returns `0.0`.

# Implementation notes
- The loop ordering, accumulation structure, and floating-point behavior
  intentionally mirror the original dimension-specific implementations
  for reproducibility.
- The helper `_call_with_axis` dynamically replaces one coordinate while
  keeping the remaining axes fixed, allowing AD-compatible evaluation.
- Derivatives are evaluated using `nth_derivative`, and therefore inherit
  its numerical and AD-related behavior.

# Limitations
- This estimator is not a rigorous error bound; it provides only a leading
  scaling model suitable for convergence fitting.
- Computational cost scales as `O(dim * length(xs)^(dim-1))`, which grows
  rapidly with dimension.
"""
function estimate_error_nd(
    f,
    a::Real,
    b::Real,
    N::Int,
    rule::Symbol;
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    aa = float(a)
    bb = float(b)
    h  = (bb - aa) / N

    xs, ws = quadrature_1d_nodes_weights(aa, bb, N, rule)

    # helper: call f with axis value replaced by x (x may be Dual)
    @inline function _call_with_axis(f, fixed::Vector{Float64}, axis::Int, x, dim::Int)
        return f(ntuple(d -> (d == axis ? x : fixed[d]), dim)...)
    end

    # ---- special boundary-difference models ----
    if _has_boundary_error_model(rule)
        if rule == :simpson13_open
            (N % 2 == 0) || error("Simpson 1/3 open-chain requires N even, got N = $N")
            (N >= 8)     || error("Simpson 1/3 open-chain requires N ≥ 8, got N = $N")
        elseif rule == :bode_open
            (N % 4 == 0) || error("Open composite Boole requires N divisible by 4, got N = $N")
            (N >= 16)    || error("Open composite Boole requires N ≥ 16 (non-overlapping end stencils), got N = $N")
        end

        p, K, dord, off = _boundary_error_params(rule)
        xL = aa + off*h
        xR = aa + (N-off)*h

        total_axes = 0.0

        fixed = Vector{Float64}(undef, dim)      # stores "other axes" values
        idx   = ones(Int, dim - 1)               # odometer for other axes

        @inbounds for axis in 1:dim
            Iaxis = 0.0

            if dim == 1
                Iaxis = nth_derivative(x -> f(x), xL, dord) -
                        nth_derivative(x -> f(x), xR, dord)
            else
                fill!(idx, 1)
                while true
                    wprod = 1.0
                    t = 1
                    for d in 1:dim
                        if d == axis
                            continue
                        end
                        i = idx[t]
                        fixed[d] = xs[i]
                        wprod *= ws[i]
                        t += 1
                    end

                    Iaxis += wprod * (
                        nth_derivative(x -> _call_with_axis(f, fixed, axis, x, dim), xL, dord) -
                        nth_derivative(x -> _call_with_axis(f, fixed, axis, x, dim), xR, dord)
                    )

                    # increment odometer on idx
                    q = dim - 1
                    while q >= 1
                        idx[q] += 1
                        if idx[q] <= length(xs)
                            break
                        else
                            idx[q] = 1
                            q -= 1
                        end
                    end
                    q == 0 && break
                end
            end

            total_axes += Iaxis
        end

        return K * h^p * total_axes
    end

    # ---- default tensor-style midpoint model ----
    m, C = _rule_params_for_tensor_error(rule)
    m == 0 && return 0.0

    xmid = (aa + bb) / 2

    total_axes = 0.0

    fixed = Vector{Float64}(undef, dim)
    idx   = ones(Int, dim - 1)

    @inbounds for axis in 1:dim
        Iaxis = 0.0

        if dim == 1
            Iaxis = nth_derivative(x -> f(x), xmid, m)
        else
            fill!(idx, 1)
            while true
                wprod = 1.0
                t = 1
                for d in 1:dim
                    if d == axis
                        continue
                    end
                    i = idx[t]
                    fixed[d] = xs[i]
                    wprod *= ws[i]
                    t += 1
                end

                Iaxis += wprod * nth_derivative(
                    x -> _call_with_axis(f, fixed, axis, x, dim),
                    xmid, m
                )

                # increment odometer on idx
                q = dim - 1
                while q >= 1
                    idx[q] += 1
                    if idx[q] <= length(xs)
                        break
                    else
                        idx[q] = 1
                        q -= 1
                    end
                end
                q == 0 && break
            end
        end

        total_axes += Iaxis
    end

    return C * (bb - aa) * h^m * total_axes
end