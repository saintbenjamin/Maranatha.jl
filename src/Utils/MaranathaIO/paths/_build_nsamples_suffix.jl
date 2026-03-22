# ============================================================================
# src/Utils/MaranathaIO/paths/_build_nsamples_suffix.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _build_nsamples_suffix(
        Ns
    ) -> String

Construct a filename-friendly suffix from subdivision counts.

# Function description
This internal helper converts a collection of subdivision counts into a compact
suffix of the form

    N_2_3_4_5

for use in default result filenames.

# Arguments
- `Ns`: Collection of subdivision counts.

# Returns
- `String`: Filename-friendly `N_...` suffix.

# Errors
- No explicit validation is performed.

# Notes
- This helper is intended for internal path-construction workflows.
"""
function _build_nsamples_suffix(
    Ns
)
    return "N_" * join(Int.(Ns), "_")
end
