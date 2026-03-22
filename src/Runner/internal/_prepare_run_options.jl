# ============================================================================
# src/Runner/internal/_prepare_run_options.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _prepare_run_options(
        nsamples::Vector{Int},
        rule,
        boundary,
        dim::Int,
        err_method::Symbol,
        use_cuda::Bool,
    ) -> NamedTuple

Prepare validated execution options for the main runner loop.

# Function description
This helper gathers the runner-level setup that should happen once before the
multi-resolution loop begins. It validates the supplied rule specification,
applies the refinement-specific same-family check for axis-wise rules, emits the
run-configuration log line, determines whether threaded-subgrid execution should
be enabled, clears derivative caches when the active path is derivative-based,
and sanitizes `nsamples` for Newton-Cotes runs.

# Arguments
- `nsamples::Vector{Int}`:
  Candidate subdivision counts requested by the caller.
- `rule`:
  Quadrature-rule specification, scalar or axis-wise.
- `boundary`:
  Boundary specification, scalar or axis-wise.
- `dim::Int`:
  Problem dimension used for rule and boundary resolution.
- `err_method::Symbol`:
  Active error-estimation mode.
- `use_cuda::Bool`:
  Whether CUDA execution was requested for the current run.

# Returns
- `NamedTuple` with fields:
  - `nsamples`: sanitized subdivision sequence actually used by the runner,
  - `threaded_subgrid`: `Bool` indicating whether threaded-subgrid execution
    should be enabled.

# Errors
- Propagates rule-validation errors from
  [`QuadratureRuleSpec._validate_rule_spec`](@ref).
- Propagates refinement-family consistency errors from
  [`QuadratureRuleSpec._common_rule_family`](@ref).
- Propagates boundary and admissibility errors from
  [`_sanitize_run_nsamples`](@ref).

# Notes
- CUDA mode disables threaded-subgrid execution in the returned option set.
- Derivative caches are cleared only for non-refinement runs.
"""
function _prepare_run_options(
    nsamples::Vector{Int},
    rule,
    boundary,
    dim::Int,
    err_method::Symbol,
    use_cuda::Bool,
)
    QuadratureRuleSpec._validate_rule_spec(rule, dim)

    if err_method === :refinement
        QuadratureRuleSpec._common_rule_family(rule, dim)
    end

    JobLoggerTools.println_benji(
        "run config: dim=$(dim), rule=$(string(rule)), boundary=$(string(boundary)), err_method=$(err_method)"
    )

    threaded_subgrid = (!use_cuda) && (Base.Threads.nthreads() > 1)

    if err_method !== :refinement
        ErrorDispatch.ErrorDispatchDerivative.clear_error_estimate_derivative_caches!()
    end

    nsamples_sane = _sanitize_run_nsamples(
        nsamples,
        rule,
        boundary,
        dim,
    )

    return (;
        nsamples = nsamples_sane,
        threaded_subgrid = threaded_subgrid,
    )
end