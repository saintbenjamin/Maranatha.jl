# ============================================================================
# src/Utils/MaranathaIO/paths/_to_axis_vector.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _to_axis_vector(x, dim::Int) -> Vector

Convert a domain specification into a concrete vector form.

# Function description

This helper converts scalar and collection-valued domain data into the vector
representation used by internal geometry and `nsamples` reconstruction code.

The current behavior is:

- `Tuple` input: returns `[x[i] for i in 1:dim]`, so the first `dim` entries
  are copied and a bounds error is raised if the tuple is shorter than `dim`.
- `AbstractVector` input: returns `collect(x)` without enforcing that the
  resulting length matches `dim`.
- Scalar input: returns `fill(x, dim)`.

# Arguments

- `x`: Domain value (scalar, tuple, or vector-like).
- `dim`: Target dimensionality used for scalar expansion and tuple indexing.

# Returns

- `Vector`: Collected axis values. Scalar input yields a length-`dim` vector;
  tuple and vector input follow the rules above.

# Errors

- May throw a bounds error if `x isa Tuple` and `length(x) < dim`.
"""
@inline function _to_axis_vector(x, dim::Int)
    if x isa Tuple
        return [x[i] for i in 1:dim]
    elseif x isa AbstractVector
        return collect(x)
    else
        return fill(x, dim)
    end
end
