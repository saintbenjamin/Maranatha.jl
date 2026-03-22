# ============================================================================
# src/Utils/MaranathaIO/paths/_filename_spec_dim.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _filename_spec_dim(a, b, rule, boundary) -> Int

Infer the effective axis count used for result-filename construction.

# Function description
This helper inspects domain bounds together with rule and boundary metadata and
returns the unique common axis count implied by any axis-wise inputs. If all
inputs are scalar-like, the returned dimension is `1`.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `Int`: Effective dimension used when expanding filename tokens.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `a` and `b` mix scalar and
  collection styles or if axis-wise inputs imply inconsistent dimensions.
"""
@inline function _filename_spec_dim(a, b, rule, boundary)::Int
    a_multi = _filename_spec_is_multi(a)
    b_multi = _filename_spec_is_multi(b)

    a_multi == b_multi || JobLoggerTools.error_benji(
        "Filename-spec mismatch: `a` and `b` must both be scalar or both be tuple/vector-like."
    )

    dims = Int[]

    if a_multi
        push!(dims, length(a))
        push!(dims, length(b))
    end
    _filename_spec_is_multi(rule)     && push!(dims, length(rule))
    _filename_spec_is_multi(boundary) && push!(dims, length(boundary))

    isempty(dims) && return 1

    dim = first(dims)
    all(==(dim), dims) || JobLoggerTools.error_benji(
        "Filename-spec mismatch: inconsistent axis counts across domain/rule/boundary."
    )

    return dim
end
