# ============================================================================
# src/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative/internal/_flatten_axiswise_error_result.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _flatten_axiswise_error_result(err_nd) -> NamedTuple

Convert the axis-wise generic derivative estimator output into the legacy flat
result layout expected by downstream fitting and serialization code.

# Function description
This helper preserves the exact per-axis decomposition in `err_nd.per_axis`,
while also constructing flat aggregate fields:

- `ks`: union of all residual orders appearing on any axis,
- `coeffs`, `derivatives`, `terms`: sums of same-`k` contributions across axes,
- `total`: sum of the flattened `terms`.

The midpoint `center` and mesh-size field `h` are forwarded unchanged from the
generic `nd` result.

# Arguments
- `err_nd`: Generic axis-wise derivative-estimator result with a `per_axis`
  field.

# Returns
- `NamedTuple`: Legacy-compatible flat result with an added `per_axis` field.

# Errors
- May throw if `err_nd.per_axis` is empty or if its entries are structurally
  inconsistent with the expected derivative-estimator result layout.

# Notes
- This helper is a compatibility bridge for the 2D/3D/4D wrappers.
- No information is discarded: the exact axis-wise pieces remain available in
  `per_axis`.
"""
@inline function _flatten_axiswise_error_result(err_nd)
    axis_results = err_nd.per_axis
    T = eltype(axis_results[1].terms)

    ks_all = Int[]
    for axis_res in axis_results
        append!(ks_all, axis_res.ks)
    end
    sort!(unique!(ks_all))

    coeffs = zeros(T, length(ks_all))
    derivatives = zeros(T, length(ks_all))
    terms = zeros(T, length(ks_all))

    for axis_res in axis_results
        for j in eachindex(axis_res.ks)
            k = axis_res.ks[j]
            i = findfirst(==(k), ks_all)
            i === nothing && continue
            coeffs[i] += axis_res.coeffs[j]
            derivatives[i] += axis_res.derivatives[j]
            terms[i] += axis_res.terms[j]
        end
    end

    return (;
        ks = ks_all,
        coeffs = coeffs,
        derivatives = derivatives,
        terms = terms,
        total = sum(terms),
        center = err_nd.center,
        h = err_nd.h,
        per_axis = axis_results,
    )
end
