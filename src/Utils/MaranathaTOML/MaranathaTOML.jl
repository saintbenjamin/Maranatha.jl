# ============================================================================
# src/Utils/MaranathaTOML/MaranathaTOML.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module MaranathaTOML

TOML parsing and validation helpers for `Maranatha.jl` run configurations.

# Module description
`Maranatha.Utils.MaranathaTOML` turns user-facing TOML files into normalized
configuration bundles, validates them, and loads integrand functions from local
Julia source files.

It supports scalar and axis-wise domain / rule / boundary specifications and
enforces the current refinement restriction that axis-wise rules must belong to
one common family.

# Main entry points
- [`parse_run_config_from_toml`](@ref)
- [`validate_run_config`](@ref)
- [`load_integrand_from_file`](@ref)
"""
module MaranathaTOML

import ..TOML
import ..DoubleFloats

import ..QuadratureBoundarySpec

# ============================================================
# Parse helpers
# ============================================================
include("parse/_is_domain_collection.jl")
include("parse/_normalize_domain_endpoint.jl")
include("parse/_real_type_symbol_to_type.jl")
include("parse/_parse_domain_scalar.jl")
include("parse/_parse_domain_endpoint.jl")
include("parse/_parse_boundary_entry.jl")
include("parse/_parse_boundary_spec.jl")
include("parse/_parse_rule_entry.jl")
include("parse/_parse_rule_spec.jl")

# ============================================================
# Validation helpers
# ============================================================
include("validate/VALID_ERR_METHODS.jl")
include("validate/_rule_family_local.jl")
include("validate/_rule_at_local.jl")
include("validate/_validate_rule_spec_local.jl")
include("validate/_validate_refinement_rule_family_local.jl")
include("validate/_domain_axis_values.jl")

# ============================================================
# Public API
# ============================================================
include("api/load_integrand_from_file.jl")
include("api/parse_run_config_from_toml.jl")
include("api/validate_run_config.jl")

end  # module MaranathaTOML
