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
        rule,
        boundary;
        dim::Int
    ) -> Float64

Perform an **multidimensional tensor-product quadrature** over the hypercube
domain ``[a,b]^{\\texttt{dim}}`` using a 1D rule specified by `rule`.

# Function description
This routine evaluates a multidimensional integral by constructing the
tensor product of a 1D quadrature rule.

The algorithm:

1. Builds 1D quadrature nodes and weights `(xs, ws)` via
   [`quadrature_1d_nodes_weights`](@ref)`(a, b, N, rule, boundary)`.
2. Iterates over all multi-indices ``(i_1, i_2, \\ldots, i_{\\texttt{dim}})`` using an
   odometer-style index update.
3. Forms the tensor-product weight

   ``w = 
   \\texttt{ws[}\\texttt{i}_\\texttt{1}\\texttt{]} \\ast
   \\texttt{ws[}\\texttt{i}_\\texttt{2}\\texttt{]} \\ast
   \\ldots \\ast 
   \\texttt{ws[}\\texttt{i}_\\texttt{dim}\\texttt{]}``.

4. Evaluates the integrand as

   ``f\\texttt{(}
   \\texttt{xs[}\\texttt{i}_\\texttt{1}\\texttt{]},\\,
   \\texttt{xs[}\\texttt{i}_\\texttt{2}\\texttt{]},\\,
   \\ldots,\\,
   \\texttt{xs[}\\texttt{i}_\\texttt{dim}\\texttt{]}
   \\texttt{)} \\,.``

5. Accumulates the weighted sum

   ``\\displaystyle{\\sum_{i_1,\\ldots,i_{\\texttt{dim}}} w \\ast f\\texttt{(}\\ldots\\texttt{)}}``.

This implementation intentionally mirrors the explicit loop ordering
and accumulation style used throughout the `Maranatha.jl` quadrature stack
to ensure reproducibility and consistent floating-point behavior.

# Arguments
- `f`: Integrand callable accepting `dim` scalar arguments.
- `a`, `b`: Domain bounds defining the hypercube ``[a,b]^\\texttt{dim}``.
- `N`: Number of subdivisions per axis used to build the 1D rule.
- `rule`: Integration rule symbol (e.g., `:simpson13_close`, `:bode_open`, etc.).
- `boundary`: Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`).
  Required for NS rules.
- `dim`: Number of dimensions (must satisfy `dim```\\ge 1``).

# Returns
- `Float64`: Numerical quadrature estimate of the integral.

# Notes
- This is a pure tensor-product construction; computational cost scales
  as ``O(\\texttt{length(xs)}^\\texttt{dim})`` and therefore grows exponentially with `dim`.
- Rule-specific constraints on `N` are enforced inside
  [`quadrature_1d_nodes_weights`](@ref).
- The integrand is called as ``f(x_1, x_2, \\ldots, x_\\texttt{dim})`` using splatting.
"""
function integrate_nd(
    f, 
    a, 
    b, 
    N, 
    rule,
    boundary;
    dim::Int
)
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))

    xs, ws = quadrature_1d_nodes_weights(a, b, N, rule, boundary)

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
        iszero(wprod) || (total += wprod * f(args...))

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