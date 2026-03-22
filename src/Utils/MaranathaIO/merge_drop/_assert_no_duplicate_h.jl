# ============================================================================
# src/Utils/MaranathaIO/merge_drop/_assert_no_duplicate_h.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _assert_no_duplicate_h(
        hs;
        atol=1e-12
    ) -> Nothing

Check that a collection of step sizes contains no duplicate values.

# Function description
This helper sorts the supplied scalar step sizes and checks adjacent values for
numerical duplication within the specified absolute tolerance.

It is used as a merge-safety check when combining datapoint result blocks.

# Arguments
- `hs`: Collection of step sizes.
- `atol`: Absolute tolerance used for duplicate detection.

# Returns
- `nothing`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if a duplicate step size is
  detected within tolerance.

# Notes
- This helper is conservative by design and rejects ambiguous overlapping
  datapoints.
"""
function _assert_no_duplicate_h(
    hs;
    atol = 1e-12,
)
    T = eltype(hs)
    atolT = convert(T, atol)

    p = sortperm(hs)
    hs_sorted = hs[p]

    for (h_prev, h_cur) in zip(hs_sorted, Iterators.drop(hs_sorted, 1))
        isapprox(h_cur, h_prev; atol = atolT, rtol = zero(T)) &&
            JobLoggerTools.error_benji(
                "Duplicate h detected during merge: h=$(h_cur)"
            )
    end

    return nothing
end
