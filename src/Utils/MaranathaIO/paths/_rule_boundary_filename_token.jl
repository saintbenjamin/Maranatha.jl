# ============================================================================
# src/Utils/MaranathaIO/paths/_rule_boundary_filename_token.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _rule_boundary_filename_token(a, b, rule, boundary) -> String

Construct the compact rule/boundary token used in saved-result filenames.

# Function description
If all inputs are scalar-like, this helper returns `"<rule>_<boundary>"`. If
any input is axis-wise, it expands the result into an axis-tagged token of the
form `"1_<rule1>_<boundary1>_2_<rule2>_<boundary2>_..."`.

# Arguments
- `a`, `b`: Domain-bound specifications used only to infer scalar vs axis-wise
  filename layout.
- `rule`: Quadrature-rule specification.
- `boundary`: Boundary specification.

# Returns
- `String`: Filename-friendly rule/boundary token.

# Errors
- Propagates dimensional-consistency errors from
  [`_filename_spec_dim`](@ref) and [`_filename_spec_at`](@ref).

# Notes
- Domain values themselves are not embedded in the returned token.
"""
@inline function _rule_boundary_filename_token(a, b, rule, boundary)::String
    dim = _filename_spec_dim(a, b, rule, boundary)

    if dim == 1
        return "$(string(rule))_$(string(boundary))"
    end

    parts = String[]
    for d in 1:dim
        push!(parts, string(d))
        push!(parts, string(_filename_spec_at(rule, d, dim)))
        push!(parts, string(_filename_spec_at(boundary, d, dim)))
    end

    return join(parts, "_")
end
