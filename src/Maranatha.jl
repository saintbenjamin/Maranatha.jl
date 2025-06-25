__precompile__(false)

module Maranatha

using ForwardDiff
using LsqFit

# === Include numerical integration rules ===
include("rules/Simpson13Rule.jl")
include("rules/Simpson38Rule.jl")
include("rules/BodeRule.jl")

using .Simpson13Rule
using .Simpson38Rule
using .BodeRule

# === Include error estimation and fitting tools ===
include("error/ErrorEstimator.jl")
include("fit/FitConvergence.jl")

using .ErrorEstimator
using .FitConvergence

export run_Maranatha, integrate_nd

"""
    run_Maranatha(integrand, a, b; dim=1, nsamples=[4,8,16], rule=:simpson13)

Evaluate a definite integral in 1–4 dimensions using Newton–Cotes quadrature  
rules and extrapolate to zero step size via least χ² fitting.

# Arguments
- `integrand::Function`: Function of 1 to 4 variables depending on `dim`.
- `a::Real`, `b::Real`: Integration bounds (applied identically across all dimensions).
- `dim::Int`: Dimensionality of the integral (1 ≤ dim ≤ 4).
- `nsamples::Vector{Int}`: List of sample sizes `N` to use, e.g., `[4, 8, 16]`.
- `rule::Symbol`: Integration rule; one of `:simpson13`, `:simpson38`, or `:bode`.

# Returns
A 3-tuple:
- `final_estimate::Float64`: Extrapolated integral value as `h → 0`.
- `fit_result::LsqFit.LsqFitResult`: Full result of the χ² fitting.
- `NamedTuple`: Data used in fitting: `(h, avg, err)` where
    - `h`: Vector of step sizes,
    - `avg`: Vector of raw integral estimates,
    - `err`: Vector of estimated integration errors.

# Example
```julia
f(x) = sin(x)
I, fit, data = run_Maranatha(f, 0.0, π; dim=1, nsamples=[4, 8, 16, 32], rule=:simpson13)
```
"""
function run_Maranatha(integrand, a, b; dim=1, nsamples=[4,8,16], rule=:simpson13)

    estimates = Float64[]     # List of integral results
    errors = Float64[]        # List of estimated errors
    hs = Float64[]            # List of step sizes

    for N in nsamples
        h = (b - a) / N
        push!(hs, h)

        # Step 1: Evaluate integral using selected rule
        I = integrate_nd(integrand, a, b, N, dim, rule)

        # Step 2: Estimate integration error
        err = estimate_error(integrand, a, b, N, dim, rule)

        push!(estimates, I)
        push!(errors, err)
    end

    # Step 3: Perform least chi-square fit to extrapolate as h → 0
    fit_result = fit_convergence(hs, estimates, errors, rule)

    return fit_result.estimate, fit_result, (; h=hs, avg=estimates, err=errors)
end

"""
    integrate_nd(integrand, a, b, N, dim, rule)

Dispatch integration method by dimension and rule.

# Arguments
- `integrand`: function of 1 to 4 variables
- `a`, `b`: integration bounds (same for all dimensions)
- `N`: number of intervals
- `dim`: dimensionality (1, 2, 3, or 4)
- `rule`: integration rule symbol (e.g. `:simpson13`, `:simpson38`, `:bode`)

# Returns
- Estimated integral value (Float64)
"""
function integrate_nd(integrand, a, b, N, dim, rule)
    if dim == 1
        return integrate_1d(integrand, a, b, N, rule)
    elseif dim == 2
        return integrate_2d(integrand, a, b, N, rule)
    elseif dim == 3
        return integrate_3d(integrand, a, b, N, rule)
    elseif dim == 4
        return integrate_4d(integrand, a, b, N, rule)
    else
        error("Dimension $dim not supported. Use dim = 1, 2, 3, or 4.")
    end
end

# 1D integration
function integrate_1d(f, a, b, N, rule)
    if rule == :simpson13
        return simpson13_rule(f, a, b, N)
    elseif rule == :simpson38
        return simpson38_rule(f, a, b, N)
    elseif rule == :bode
        return bode_rule(f, a, b, N)
    else
        error("Unknown integration rule: $rule")
    end
end

# 2D integration (apply rule along each axis)
function integrate_2d(f, a, b, N, rule)
    h = (b - a) / N
    xs = range(a, b; length=N+1)
    ys = range(a, b; length=N+1)
    total = 0.0
    for x in xs
        fx(y) = f(x, y)
        total += integrate_1d(fx, a, b, N, rule)
    end
    return total * h / (b - a)
end

# 3D integration
function integrate_3d(f, a, b, N, rule)
    h = (b - a) / N
    xs = range(a, b; length=N+1)
    ys = range(a, b; length=N+1)
    zs = range(a, b; length=N+1)
    total = 0.0
    for x in xs, y in ys
        fxy(z) = f(x, y, z)
        total += integrate_1d(fxy, a, b, N, rule)
    end
    return total * h^2 / (b - a)^2
end

# 4D integration
function integrate_4d(f, a, b, N, rule)
    h = (b - a) / N
    xs = range(a, b; length=N+1)
    ys = range(a, b; length=N+1)
    zs = range(a, b; length=N+1)
    ts = range(a, b; length=N+1)
    total = 0.0
    for x in xs, y in ys, z in zs
        fxyz(t) = f(x, y, z, t)
        total += integrate_1d(fxyz, a, b, N, rule)
    end
    return total * h^3 / (b - a)^3
end

end  # module Maranatha