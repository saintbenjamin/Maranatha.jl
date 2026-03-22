# ============================================================================
# src/Utils/MaranathaTOML/validate/_domain_axis_values.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _domain_axis_values(
        x, 
        dim::Int
    ) -> Vector

Expand a domain endpoint into explicit per-axis values of length `dim`.

# Function description

This helper converts a scalar or collection-valued endpoint into the
axis-by-axis representation used by [`validate_run_config`](@ref).

Behavior depends on `x`:

- if `x isa Tuple`, the tuple length must equal `dim` and the elements are
  copied into a new vector;
- if `x isa AbstractVector`, the vector length must equal `dim` and the
  values are collected into a new vector;
- otherwise, `x` is treated as a scalar and replicated `dim` times.

# Arguments

- `x`: Domain endpoint (scalar, tuple, or vector-like).
- `dim`: Expected number of dimensions.

# Returns

- `Vector`: A length-`dim` vector of axis endpoint values.

# Errors

- Throws an error if a tuple or vector input does not have length `dim`.
"""
@inline function _domain_axis_values(
    x, 
    dim::Int
)
    if x isa Tuple
        length(x) == dim || error(
            "Domain tuple length mismatch: expected dim=$(dim), got length=$(length(x))."
        )
        return [x[i] for i in 1:dim]
    elseif x isa AbstractVector
        length(x) == dim || error(
            "Domain vector length mismatch: expected dim=$(dim), got length=$(length(x))."
        )
        return collect(x)
    else
        return fill(x, dim)
    end
end
