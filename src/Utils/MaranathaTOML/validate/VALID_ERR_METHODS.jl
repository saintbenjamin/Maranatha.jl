# ============================================================================
# src/Utils/MaranathaTOML/validate/VALID_ERR_METHODS.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    VALID_ERR_METHODS :: Set{Symbol}

Supported `err_method` selectors for TOML-driven runs.

# Description
This constant contains the symbol values currently accepted for
`cfg.err_method` by [`validate_run_config`](@ref).

The set is used only for configuration validation. It does not perform
backend dispatch by itself.

# Notes
- Supported values are `:refinement`, `:forwarddiff`, `:taylorseries`,
  and `:enzyme`.
- Derivative-based methods and `:refinement` use different `nerr_terms`
  validation rules; see [`validate_run_config`](@ref).
"""
const VALID_ERR_METHODS = Set([
    :refinement,
    :forwarddiff,
    :taylorseries,
    :enzyme,
])
