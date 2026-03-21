"""
    module QuadratureRuleSpec

Internal helpers for validating and normalizing quadrature-rule specifications.

# Module description
`QuadratureRuleSpec` centralizes the handling of rule specifications that may be given
either as:

- a single rule symbol shared by all axes, or
- a tuple / vector of rule symbols with one entry per axis.

The helpers in this module are used throughout the quadrature, error-estimate,
and runner layers so that axis-wise rule handling is implemented consistently.
"""
module QuadratureRuleSpec

import ..JobLoggerTools
import ..Quadrature.NewtonCotes
import ..Quadrature.Gauss
import ..Quadrature.BSpline

"""
    _rule_family(rule::Symbol) -> Symbol

Classify a scalar quadrature-rule symbol by family.

# Function description
This helper maps one resolved scalar quadrature-rule symbol to its enclosing
rule family. It is used internally after axis-wise specifications have already
been reduced to one local rule entry.

# Arguments
- `rule::Symbol`:
  Scalar quadrature-rule symbol to classify.

# Returns
- `Symbol`:
  One of `:newton_cotes`, `:gauss`, or `:bspline`.

# Errors
- Throws via [`JobLoggerTools.error_benji`](@ref) if `rule` is not a supported
  quadrature-rule symbol.

# Notes
- This helper accepts only scalar rule symbols, not axis-wise rule
  specifications.
"""
@inline function _rule_family(rule::Symbol)::Symbol
    if NewtonCotes._is_newton_cotes_rule(rule)
        return :newton_cotes
    elseif Gauss._is_gauss_rule(rule)
        return :gauss
    elseif BSpline._is_bspline_rule(rule)
        return :bspline
    else
        JobLoggerTools.error_benji("unsupported quadrature rule=$rule")
    end
end

"""
    _rule_at(rule, d::Int, dim::Int) -> Symbol

Resolve the scalar quadrature rule used on axis `d`.

# Function description
If `rule` is a scalar symbol, it is shared across all axes and returned
unchanged. If `rule` is a tuple or vector, this helper validates that its
length matches `dim`, checks that `rule[d]` is a supported rule symbol, and
returns that axis-local entry.

# Arguments
- `rule`:
  Quadrature-rule specification. This may be either a scalar rule symbol shared
  across all axes or a tuple/vector of rule symbols of length `dim`.
- `d::Int`:
  Axis index to resolve.
- `dim::Int`:
  Problem dimension used when validating axis-wise rule specifications.

# Returns
- `Symbol`:
  Scalar rule symbol used on axis `d`.

# Errors
- Throws `ArgumentError` if an axis-wise rule specification has the wrong
  length or contains a non-symbol entry.
- Propagates unsupported-rule errors from [`_rule_family`](@ref).

# Notes
- Scalar rule symbols are returned unchanged after validation.
"""
@inline function _rule_at(rule::Symbol, d::Int, dim::Int)::Symbol
    _rule_family(rule)
    return rule
end

@inline function _rule_at(rule::Tuple, d::Int, dim::Int)::Symbol
    length(rule) == dim || throw(ArgumentError("length(rule) must equal dim"))
    rl = rule[d]
    rl isa Symbol || throw(ArgumentError("rule[$d] must be a Symbol"))
    _rule_family(rl)
    return rl
end

@inline function _rule_at(rule::AbstractVector, d::Int, dim::Int)::Symbol
    length(rule) == dim || throw(ArgumentError("length(rule) must equal dim"))
    rl = rule[d]
    rl isa Symbol || throw(ArgumentError("rule[$d] must be a Symbol"))
    _rule_family(rl)
    return rl
end

"""
    _validate_rule_spec(rule, dim::Int) -> Nothing

Validate that `rule` is a well-formed quadrature-rule specification for
dimension `dim`.

# Function description
This helper accepts either a scalar rule symbol shared across all axes or an
axis-wise tuple/vector of rule symbols of length `dim`. Every resolved per-axis
entry is checked against the supported quadrature families.

# Arguments
- `rule`:
  Quadrature-rule specification to validate.
- `dim::Int`:
  Problem dimension that the rule specification must be compatible with.

# Returns
- `nothing`

# Errors
- Throws `ArgumentError` if `dim < 1` or if an axis-wise rule specification has
  the wrong length or invalid element types.
- Propagates unsupported-rule errors from [`_rule_family`](@ref).

# Notes
- Successful return means that `rule` can safely be resolved by
  [`_rule_at`](@ref) on every axis `1:dim`.
"""
@inline function _validate_rule_spec(rule, dim::Int)::Nothing
    dim >= 1 || throw(ArgumentError("dim must be ≥ 1"))
    for d in 1:dim
        _rule_at(rule, d, dim)
    end
    return nothing
end

"""
    _has_axiswise_rule_spec(rule, dim::Int) -> Bool

Return `true` if `rule` is an explicit per-axis rule specification.

# Function description
This helper first validates `rule` against `dim`, then distinguishes scalar
shared rules from explicit tuple/vector axis-wise rule specifications.

# Arguments
- `rule`:
  Quadrature-rule specification to test.
- `dim::Int`:
  Problem dimension used when validating `rule`.

# Returns
- `Bool`:
  `true` when `rule` is an explicit axis-wise tuple/vector specification and
  `dim > 1`, otherwise `false`.

# Errors
- Propagates validation errors from [`_validate_rule_spec`](@ref).

# Notes
- Scalar rule symbols shared across all axes return `false`.
- Tuple / vector rule specifications return `true` after validation, provided
  `dim > 1`.
"""
@inline function _has_axiswise_rule_spec(rule, dim::Int)::Bool
    _validate_rule_spec(rule, dim)
    return (rule isa Tuple || rule isa AbstractVector) && dim > 1
end

"""
    _common_rule_family(rule, dim::Int) -> Symbol

Return the common quadrature family used by all axes in `rule`.

# Function description
This helper is mainly used by refinement-based logic, where all per-axis rule
entries must belong to one quadrature family. Scalar rules trivially satisfy
this condition. For axis-wise tuple/vector specifications, every axis-local
rule is validated and checked for family consistency.

# Arguments
- `rule`:
  Quadrature-rule specification, scalar or axis-wise.
- `dim::Int`:
  Problem dimension used when resolving axis-local rule entries.

# Returns
- `Symbol`:
  The unique common family tag shared by all axes.

# Errors
- Throws `ArgumentError` if the resolved per-axis rules do not all belong to
  the same family.
- Propagates validation errors from [`_validate_rule_spec`](@ref).

# Notes
- This helper is primarily used by refinement-based logic, where mixed
  rule-family axis-wise specifications are intentionally rejected.
"""
@inline function _common_rule_family(rule, dim::Int)::Symbol
    _validate_rule_spec(rule, dim)

    fam = _rule_family(_rule_at(rule, 1, dim))
    for d in 2:dim
        fam_d = _rule_family(_rule_at(rule, d, dim))
        fam_d === fam || throw(ArgumentError(
            "axis-wise rule spec for refinement must use a single rule family (got rule=$rule)"
        ))
    end

    return fam
end

"""
    _rule_spec_string(rule) -> String

Convert a rule specification into a compact underscore-joined string.

# Function description
Scalar rule symbols are converted directly. Tuple and vector rule
specifications are flattened and joined with underscores. This helper is used
for human-readable identifiers such as filenames and summary labels.

# Arguments
- `rule`:
  Quadrature-rule specification to stringify.

# Returns
- `String`:
  Compact string representation of `rule`.

# Errors
- Does not perform rule-family validation on its own. Invalid entries are
  stringified as provided.

# Notes
- Axis-wise tuple/vector rule specifications are joined with underscores in
  axis order.
"""
@inline function _rule_spec_string(rule)::String
    if rule isa Symbol
        return String(rule)
    elseif rule isa Tuple || rule isa AbstractVector
        return join(String.(collect(rule)), "_")
    else
        return string(rule)
    end
end

end  # module QuadratureRuleSpec
