# ============================================================================
# src/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid/internal/_quadrature_nd_serial_from_nodes.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

function _quadrature_nd_serial_from_nodes(
    f,
    xs_list::Vector{Vector{T}},
    ws_list::Vector{Vector{T}},
    lens::AbstractVector{<:Integer},
    dim::Int,
) where {T}
    idx = ones(Int, dim)
    args = Vector{T}(undef, dim)
    total = zero(T)

    while true
        wprod = one(T)
        for d in 1:dim
            ii = idx[d]
            args[d] = xs_list[d][ii]
            wprod *= ws_list[d][ii]
        end

        iszero(wprod) || (total += wprod * f(args...))

        carry_dim = dim
        while carry_dim >= 1
            idx[carry_dim] += 1
            if idx[carry_dim] <= lens[carry_dim]
                break
            else
                idx[carry_dim] = 1
                carry_dim -= 1
            end
        end

        carry_dim == 0 && break
    end

    return total
end
