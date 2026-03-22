# ============================================================================
# src/ErrorEstimate/AutoDerivative/AutoDerivativeJet/internal/_derivative_values_for_ks.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    _derivative_values_for_ks(
        jet_fun,
        backend_tag::Symbol,
        g,
        x0,
        ks::AbstractVector{<:Integer};
        real_type = nothing,
    ) -> AbstractVector{<:Real}

Return selected derivative values of `g` at `x0` for the derivative orders
listed in `ks`.

# Function description
This helper computes a derivative jet of `g` up to the maximum order appearing
in `ks`, then extracts only the requested derivative orders and returns them as
a dense vector in the active scalar type.

More precisely, if

```julia
ks = [k₁, k₂, ..., k_m]
```

then the returned vector is

```julia
[g^(k₁)(x0), g^(k₂)(x0), ..., g^(k_m)(x0)]
```

with each entry obtained from the shared jet produced by
[`derivative_jet`](@ref). This is useful when several specific derivative
orders are needed at the same point, since one jet can serve all of them.

# Arguments
- `jet_fun`:
  Backend-specific derivative-jet routine, typically returned by
  [`resolve_derivative_jet_backend`](@ref).
- `backend_tag::Symbol`:
  Canonical backend tag used in the jet cache key.
- `g`:
  Scalar callable.
- `x0`:
  Evaluation point.
- `ks::AbstractVector{<:Integer}`:
  Requested derivative orders.

# Keyword arguments
- `real_type = nothing`:
  Optional scalar type used for jet construction and output conversion.
  If `nothing`, the function uses `typeof(float(x0))`.

# Returns
- `AbstractVector{<:Real}`:
  A vector containing the requested derivative values in the same order as `ks`,
  converted to the active scalar type.

# Notes
- If `ks` is empty, the function returns an empty vector of the active scalar type.
- The derivative jet is computed only up to `maximum(ks)`.
- Since extraction is performed from a shared jet, this helper is typically more
  efficient than requesting each derivative separately.
"""
@inline function _derivative_values_for_ks(
    jet_fun,
    backend_tag::Symbol,
    g,
    x0,
    ks::AbstractVector{<:Integer};
    real_type = nothing,
)
    T = isnothing(real_type) ? typeof(float(x0)) : real_type
    isempty(ks) && return T[]

    nmax = maximum(ks)

    jet = derivative_jet(
        jet_fun,
        backend_tag,
        g,
        x0,
        nmax;
        real_type = T,
    )

    vals = Vector{T}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        k = ks[i]
        vals[i] = jet[k + 1]
    end

    return vals
end
