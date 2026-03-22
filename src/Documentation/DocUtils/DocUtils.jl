# ============================================================================
# src/Documentation/DocUtils/DocUtils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DocUtils

Shared documentation-output helpers for `Maranatha.jl`.

# Module description
`Maranatha.Documentation.DocUtils` contains small helper functions reused by
plotting and reporting code. These utilities focus on name normalization and
compact axis-wise filename tokens derived from domain, rule, and boundary
metadata.

# Main entry points
- [`_split_report_name`](@ref)
- [`_rule_boundary_filename_token`](@ref)
"""
module DocUtils

include("names/_split_report_name.jl")
include("tokens/_report_name_is_multi.jl")
include("tokens/_report_name_cfg_dim.jl")
include("tokens/_report_name_cfg_at.jl")
include("tokens/_rule_boundary_filename_token.jl")

end  # module DocUtils
