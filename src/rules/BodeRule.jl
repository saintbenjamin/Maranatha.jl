module BodeRule

# Bode's Rule (closed Newton-Cotes, 4 points)
# Integration over 1D interval using 4-point Bode rule

"""
    bode_1d(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 1D function `f(x)` over `[a, b]` using Bode's rule  
(4-point closed Newton–Cotes). The number of intervals `N` must be divisible by 4.

# Arguments
- `f`: Function of a single variable, `f(x)`
- `a`, `b`: Integration bounds
- `N`: Number of subdivisions (must satisfy `N % 4 == 0`)

# Returns
- Estimated integral of `f` over `[a, b]` using Bode's rule
"""
function bode_1d(f::Function, a::Float64, b::Float64, N::Int)
    if N % 4 != 0
        error("Bode's rule requires N divisible by 4")
    end

    h = (b - a) / N
    sum = 0.0

    for i in 0:(N ÷ 4 - 1)
        x0 = a + 4i * h
        x1 = x0 + h
        x2 = x1 + h
        x3 = x2 + h
        x4 = x3 + h

        sum += 7*f(x0) + 32*f(x1) + 12*f(x2) + 32*f(x3) + 7*f(x4)
    end

    return (2h / 45) * sum
end

# Multi-dimensional extension via repeated 1D applications

"""
    bode_nd(f::Function, a::Float64, b::Float64, N::Int, dim::Int) -> Float64

Multi-dimensional integration using nested applications of 1D Bode’s rule.

# Arguments
- `f`: Function of `dim` variables
- `a`, `b`: Integration bounds (same for all axes)
- `N`: Number of intervals per axis (must be divisible by 4)
- `dim`: Dimension (1, 2, 3, or 4)

# Returns
- Estimated integral value over the `dim`-dimensional hypercube
"""
function bode_nd(f::Function, a::Float64, b::Float64, N::Int, dim::Int)
    if dim == 1
        return bode_1d(f, a, b, N)
    elseif dim == 2
        return bode_2d(f, a, b, N)
    elseif dim == 3
        return bode_3d(f, a, b, N)
    elseif dim == 4
        return bode_4d(f, a, b, N)
    else
        error("Unsupported dimension: $dim")
    end
end

"""
    bode_2d(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 2D function `f(x, y)` over `[a, b] × [a, b]`  
using iterated 1D Bode’s rule along both axes.

# Returns
- Estimated double integral of `f(x, y)`
"""
function bode_2d(f::Function, a::Float64, b::Float64, N::Int)
    bode_1d_y(y -> bode_1d(x -> f(x, y), a, b, N), a, b, N)
end

"""
    bode_3d(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 3D function `f(x, y, z)` over a cube  
`[a, b] × [a, b] × [a, b]` using nested Bode’s rule.

# Returns
- Estimated triple integral of `f(x, y, z)`
"""
function bode_3d(f::Function, a::Float64, b::Float64, N::Int)
    bode_1d_z(z -> bode_2d((x, y) -> f(x, y, z), a, b, N), a, b, N)
end

"""
    bode_4d(f::Function, a::Float64, b::Float64, N::Int) -> Float64

Numerically integrate a 4D function `f(x, y, z, t)` over a hypercube  
`[a, b]^4` using repeated 1D Bode’s rule.

# Returns
- Estimated 4D integral of `f(x, y, z, t)`
"""
function bode_4d(f::Function, a::Float64, b::Float64, N::Int)
    bode_1d_t(t -> bode_3d((x, y, z) -> f(x, y, z, t), a, b, N), a, b, N)
end

# Aliases for reuse
const bode_1d_y = bode_1d
const bode_1d_z = bode_1d
const bode_1d_t = bode_1d

export bode_nd

end  # module BodeRule