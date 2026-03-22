# ============================================================================
# src/Runner/internal/_sanitize_run_nsamples.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _sanitize_run_nsamples(
        nsamples::Vector{Int},
        rule,
        boundary,
        dim::Int = 1,
    ) -> Vector{Int}

Sanitize a candidate subdivision sequence for Newton-Cotes composite rules.

# Function description
Newton-Cotes composite formulas do not accept arbitrary subdivision counts.
For a given local node count `p` and boundary pattern `boundary`, valid values
must satisfy the tiling constraint

```math
N_{\\mathrm{sub}} = w_L + m (p-1) + w_R,
```

where `w_L` and `w_R` are the boundary block widths and `m ≥ 0` is an integer.

When `rule` and `boundary` are axis-wise specifications, this helper computes a
single subdivision sequence that is simultaneously valid for every
Newton-Cotes-active axis. Axes that use non-Newton-Cotes rule families are
ignored for this admissibility check.

This helper transforms an arbitrary input sequence into a valid sequence that:

* preserves the original length,
* forms a valid arithmetic progression with step `(p - 1)`,
* starts from the nearest admissible value not exceeding the first input
  element, or from the smallest admissible value if none is smaller.

If `rule` is not a Newton-Cotes rule, the input is returned unchanged.

# Arguments

* `nsamples::Vector{Int}`
  Candidate subdivision counts supplied by the caller.

* `rule`
  Quadrature rule specification. This may be either a scalar rule symbol shared
  across all axes or a tuple / vector of rule symbols of length `dim`.

* `boundary`
  Boundary specification. This may be either a scalar boundary symbol shared
  across all axes or a tuple / vector of boundary symbols of length `dim`.

* `dim::Int = 1`
  Problem dimension used when resolving axis-wise `rule` / `boundary`
  specifications.

# Returns

* `Vector{Int}`
  A corrected subdivision sequence compatible with the Newton-Cotes composite
  tiling constraint. The returned vector has the same length as `nsamples`.

# Errors

* Propagates validation errors from rule and boundary specification checks, and
  from the Newton-Cotes helper routines used to determine admissible
  subdivision counts.
* Does not throw for inadmissible subdivision counts; instead, it adjusts them.

# Notes

* This helper is intended for internal use by runner-level components that
  accept user-supplied subdivision arrays.
* A warning is emitted if the sequence is modified.
* The resulting sequence always represents a monotone refinement ladder whose
  common step is the least common multiple of all active Newton-Cotes block
  widths.
"""
function _sanitize_run_nsamples(
    nsamples::Vector{Int},
    rule,
    boundary,
    dim::Int = 1,
)::Vector{Int}

    isempty(nsamples) && return nsamples

    QuadratureBoundarySpec._validate_boundary_spec(boundary, dim)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    rule_axes = [QuadratureRuleSpec._rule_at(rule, d, dim) for d in 1:dim]
    nc_axes = [d for d in 1:dim if NewtonCotes._is_newton_cotes_rule(rule_axes[d])]

    isempty(nc_axes) && return nsamples

    function _is_valid_common_N(Ncand::Int)::Bool
        for d in nc_axes
            rd = rule_axes[d]
            bd = QuadratureBoundarySpec._boundary_at(boundary, d, dim)
            p = NewtonCotes._parse_newton_p(rd)
            Nd = NewtonCotes._nearest_valid_Nsub(p, bd, Ncand)
            Nd == Ncand || return false
        end
        return true
    end

    steps = Int[Quadrature.NewtonCotes._parse_newton_p(rule_axes[d]) - 1 for d in nc_axes]
    step = foldl(lcm, steps; init = 1)

    N0 = first(nsamples)
    start = nothing

    for Ncand in N0:-1:1
        if _is_valid_common_N(Ncand)
            start = Ncand
            break
        end
    end

    if isnothing(start)
        Ncand = 1
        while true
            if _is_valid_common_N(Ncand)
                start = Ncand
                break
            end
            Ncand += 1
        end
    end

    newN = [start + (i - 1) * step for i in 1:length(nsamples)]

    if newN != nsamples
        JobLoggerTools.warn_benji(
            "nsamples corrected for rule=$(rule), boundary=$(boundary), dim=$(dim)\n" *
            "input = $(nsamples)\n" *
            "using = $(newN)"
        )
    end

    return newN
end