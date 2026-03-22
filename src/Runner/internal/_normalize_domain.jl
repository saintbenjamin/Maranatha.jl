# ============================================================================
# src/Runner/internal/_normalize_domain.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _normalize_domain(
        a,
        b,
        dim::Int,
        T,
    ) -> NamedTuple

Normalize a runner domain specification into the active scalar type.

# Function description
This helper converts the user-supplied bounds `a` and `b` into the scalar type
`T` selected for the current run and determines whether the domain should be
treated as a scalar hypercube or as an axis-wise rectangular box.

If `a` and `b` are scalar-like, scalar values of type `T` are returned. If they
are tuple/vector-like, they are converted componentwise and returned in the
normalized form expected by downstream quadrature and error-estimation code.

# Arguments
- `a`:
  Lower-bound specification supplied to the runner.
- `b`:
  Upper-bound specification supplied to the runner.
- `dim::Int`:
  Declared problem dimension.
- `T`:
  Scalar type to which the bounds should be converted.

# Returns
- `NamedTuple` with fields:
  - `is_rect_domain`: `Bool` indicating whether the domain is axis-wise,
  - `aT`: normalized lower-bound specification,
  - `bT`: normalized upper-bound specification.

# Errors
- Throws `ArgumentError` if tuple/vector bounds are supplied and
  `length(a) != dim` or `length(b) != dim`.
- Propagates conversion errors if a bound component cannot be converted to `T`.

# Notes
- Scalar bounds correspond to the hypercube convention `[a,b]^dim`.
- Axis-wise bounds correspond to the rectangular-box convention
  `[a_1,b_1] × ⋯ × [a_dim,b_dim]`.
"""
function _normalize_domain(
    a,
    b,
    dim::Int,
    T,
)
    is_rect_domain = a isa AbstractVector || a isa Tuple

    if !is_rect_domain
        return (;
            is_rect_domain = false,
            aT = convert(T, a),
            bT = convert(T, b),
        )
    end

    length(a) == dim || throw(ArgumentError("length(a) must equal dim"))
    length(b) == dim || throw(ArgumentError("length(b) must equal dim"))

    aT = a isa Tuple ?
        ntuple(i -> convert(T, a[i]), dim) :
        T[convert(T, a[i]) for i in 1:dim]

    bT = b isa Tuple ?
        ntuple(i -> convert(T, b[i]), dim) :
        T[convert(T, b[i]) for i in 1:dim]

    return (;
        is_rect_domain = true,
        aT = aT,
        bT = bT,
    )
end