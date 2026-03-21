# ============================================================================
# src/Documentation/Reporter/_report_cfg_dim.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _report_cfg_dim(a, b, rule, boundary) -> Int

Infer the effective reporting dimension from domain, rule, and boundary
metadata.

# Function description
This helper returns the common axis count implied by any axis-wise inputs used
in run-configuration tables. Scalar shared inputs contribute no dimension. If
all inputs are scalar-like, the returned dimension is `1`.

# Arguments
- `a`, `b`: Domain-bound specifications.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `Int`: Effective dimension used when expanding run-configuration rows.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `a` and `b` mix scalar and
  collection styles or if axis-wise inputs imply inconsistent dimensions.
"""
@inline function _report_cfg_dim(a, b, rule, boundary)::Int
    a_multi = _report_cfg_is_multi(a)
    b_multi = _report_cfg_is_multi(b)

    if a_multi != b_multi
        JobLoggerTools.error_benji(
            "Run-configuration formatting mismatch: `a` and `b` must both be scalar or both be tuple/vector-like."
        )
    end

    dims = Int[]

    if a_multi
        push!(dims, length(a))
        push!(dims, length(b))
    end
    _report_cfg_is_multi(rule)     && push!(dims, length(rule))
    _report_cfg_is_multi(boundary) && push!(dims, length(boundary))

    isempty(dims) && return 1

    dim = first(dims)
    all(==(dim), dims) || JobLoggerTools.error_benji(
        "Run-configuration formatting mismatch: inconsistent axis counts across domain/rule/boundary."
    )

    return dim
end