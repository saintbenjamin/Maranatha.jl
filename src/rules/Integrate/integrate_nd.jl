# ============================================================================
# src/rules/Integrate/integrate_nd.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    integrate_nd(
        f,
        a,
        b,
        N,
        rule;
        dim::Int
    ) -> Float64

Perform an **N-dimensional tensor-product quadrature** over the hypercube
domain `[a,b]^dim` using a 1D rule specified by `rule`.

# Function description
This routine evaluates a multidimensional integral by constructing the
tensor product of a 1D quadrature rule.

The algorithm:

1. Builds 1D quadrature nodes and weights `(xs, ws)` via
   `quadrature_1d_nodes_weights(a, b, N, rule)`.
2. Iterates over all multi-indices `(i₁, i₂, …, i_dim)` using an
   odometer-style index update.
3. Forms the tensor-product weight

   `w = ws[i₁] * ws[i₂] * ... * ws[i_dim]`.

4. Evaluates the integrand as

   `f(xs[i₁], xs[i₂], ..., xs[i_dim])`.

5. Accumulates the weighted sum

   `∑ w * f(...)`.

This implementation intentionally mirrors the explicit loop ordering
and accumulation style used throughout the Maranatha integration stack
to ensure reproducibility and consistent floating-point behavior.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Domain bounds defining the hypercube `[a,b]^dim`.
- `N`: Number of subdivisions per axis used to build the 1D rule.
- `rule`: Integration rule symbol (e.g., `:simpson13`, `:simpson38`,
  `:bode`, `:simpson13_open`, `:bode_open`, etc.).
- `dim`: Number of dimensions (must satisfy `dim ≥ 1`).

# Returns
- `Float64`: Numerical quadrature estimate of the integral.

# Notes
- This is a pure tensor-product construction; computational cost scales
  as `O(length(xs)^dim)` and therefore grows exponentially with `dim`.
- Rule-specific constraints on `N` are enforced inside
  `quadrature_1d_nodes_weights`.
- The integrand is called as `f(x1, x2, ..., x_dim)` using splatting.

"""
function integrate_nd(
    f, 
    a, 
    b, 
    N, 
    rule; 
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    xs, ws = quadrature_1d_nodes_weights(a, b, N, rule)

    # Multi-index over axes (1-based)
    idx = ones(Int, dim)

    total = 0.0
    args = Vector{Float64}(undef, dim)

    @inbounds while true
        wprod = 1.0
        for d in 1:dim
            i = idx[d]
            args[d] = xs[i]
            wprod *= ws[i]
        end

        # Call f(x1, x2, ..., x_dim)
        total += wprod * f(args...)

        # Increment odometer-style index
        d = dim
        while d >= 1
            idx[d] += 1
            if idx[d] <= length(xs)
                break
            else
                idx[d] = 1
                d -= 1
            end
        end
        d == 0 && break
    end

    return total
end