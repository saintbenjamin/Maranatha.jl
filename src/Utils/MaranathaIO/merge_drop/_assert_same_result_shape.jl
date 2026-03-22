# ============================================================================
# src/Utils/MaranathaIO/merge_drop/_assert_same_result_shape.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _assert_same_result_shape(
        results
    ) -> Nothing

Check that multiple result objects are mutually compatible for merging.

# Function description
This helper verifies that a collection of result objects shares the same global
integration metadata before datapoint arrays are merged.

The merge is considered valid only when all inputs correspond to the same
overall computational setup and differ only in their sampled datapoints.

# Arguments
- `results`: Collection of result objects to compare.

# Returns
- `nothing`

# Errors
- Throws (via [`JobLoggerTools.error_benji`](@ref)) if fewer than two results
  are supplied.
- Throws if any required metadata field differs between result objects.

# Notes
- This helper checks metadata compatibility only; it cannot prove semantic
  identity of the original integrand.
"""
function _assert_same_result_shape(
    results
)
    length(results) >= 2 || JobLoggerTools.error_benji(
        "_assert_same_result_shape requires at least 2 results"
    )

    ref = results[1]

    for (k, res) in enumerate(results[2:end])
        i = k + 1

        res.a == ref.a || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: a differs ($(res.a) != $(ref.a))"
        )
        res.b == ref.b || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: b differs ($(res.b) != $(ref.b))"
        )
        res.dim == ref.dim || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: dim differs ($(res.dim) != $(ref.dim))"
        )
        res.rule == ref.rule || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: rule differs ($(res.rule) != $(ref.rule))"
        )
        res.boundary == ref.boundary || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: boundary differs ($(res.boundary) != $(ref.boundary))"
        )
        res.err_method == ref.err_method || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: err_method differs ($(res.err_method) != $(ref.err_method))"
        )
        res.nerr_terms == ref.nerr_terms || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: nerr_terms differs ($(res.nerr_terms) != $(ref.nerr_terms))"
        )
        res.fit_terms == ref.fit_terms || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: fit_terms differs ($(res.fit_terms) != $(ref.fit_terms))"
        )
        res.ff_shift == ref.ff_shift || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: ff_shift differs ($(res.ff_shift) != $(ref.ff_shift))"
        )
        res.use_error_jet == ref.use_error_jet || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: use_error_jet differs ($(res.use_error_jet) != $(ref.use_error_jet))"
        )
        getproperty(res, :use_cuda) == getproperty(ref, :use_cuda) || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: use_cuda differs ($(getproperty(res, :use_cuda)) != $(getproperty(ref, :use_cuda)))"
        )
        string(getproperty(res, :real_type)) == string(getproperty(ref, :real_type)) || JobLoggerTools.error_benji(
            "Merge mismatch at result $i: real_type differs ($(getproperty(res, :real_type)) != $(getproperty(ref, :real_type)))"
        )
    end

    return nothing
end
