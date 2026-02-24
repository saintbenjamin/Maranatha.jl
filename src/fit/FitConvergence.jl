# ============================================================================
# src/fit/FitConvergence.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module FitConvergence

using LinearAlgebra
using Statistics
using Printf
using ..AvgErrFormatter

export fit_convergence, print_fit_result

"""
    fit_convergence(
        hs, 
        estimates, 
        errors, 
        rule::Symbol; 
        nterms::Int=2
    )

Perform a weighted linear least-χ² extrapolation for the integral estimate
in the zero-step limit (`h → 0`).

# Function description
This routine constructs a convergence model that is linear in its parameters and
solves it using weighted least squares (WLS). The rule-dependent leading power
`p` is inferred from `rule`. The step size is taken as `h = (b-a)/N` (provided
by the caller via `hs`).

The convergence model is selected by the number of basis terms `nterms`
(including the constant term). The design matrix is constructed as:

- column 1: `h^0` (constant term)
- columns 2..nterms: `h^(p), h^(p+2), h^(p+4), ...`

Equivalently, the fitted form is:

`I(h) = I0 + C1*h^p + C2*h^(p+2) + C3*h^(p+4) + ...`

The implementation preserves the original solve order and arithmetic. In
particular:
- the weights are constructed as `W = Diagonal(1 ./ σ)`,
- the WLS system is solved via `params = (W*X) \\ (W*y)`,
- the covariance is computed as `Cov = inv(X' * (W^2) * X)`.

# Arguments
- `hs`: Vector-like collection of step sizes `h`.
- `estimates`: Vector-like collection of raw integral estimates `I(h)`.
- `errors`: Vector-like collection of error estimates associated with `I(h)`.
  Non-positive entries are replaced by `1e-8` in the internal `σ` vector.
- `rule`: Quadrature rule symbol used to select the leading convergence power `p`.

# Keyword arguments
- `nterms::Int=2`: Number of basis terms in the convergence model (including the
  constant term). Must satisfy `nterms ≥ 2`.

# Returns
A `NamedTuple` with the following fields:
- `estimate::Float64`: Extrapolated value `I0 = I(h→0)`.
- `estimate_error::Float64`: One-sigma uncertainty for `I0`, taken from the
  covariance diagonal.
- `params`: Fitted parameter vector `[I0, C1, C2, ...]`.
- `param_errors`: One-sigma uncertainties for `params`.
- `cov`: Parameter covariance matrix.
- `chisq::Float64`: χ² value, `Σ (resid/σ)^2`.
- `redchisq::Float64`: Reduced χ², `chisq / dof`.
- `dof::Int`: Degrees of freedom, `length(y) - length(params)`.

# Errors
- Throws an error if `rule` is not recognized.
- Throws an error if `nterms < 2`.
- Note: If `dof == 0`, `redchisq` will be `Inf`/`NaN` depending on `chisq`.
"""
function fit_convergence(
    hs, 
    estimates, 
    errors, 
    rule::Symbol; 
    nterms::Int=2
)

    p =
        rule == :simpson13_close ? 4 :
        rule == :simpson13_open  ? 4 :
        rule == :simpson38_close ? 4 :
        rule == :simpson38_open  ? 4 :
        rule == :bode_close      ? 6 :
        rule == :bode_open       ? 6 :
        error("Unknown rule")

    h = collect(float.(hs))
    y = collect(float.(estimates))
    σ = map(e -> e > 0 ? float(e) : 1e-8, errors)

    N = length(h)

    if nterms < 2
        error("nterms must be >= 2 (got $nterms)")
    end

    # Design matrix:
    #   col 1: 1
    #   col k (k>=2): h^(p + 2*(k-2))  => h^p, h^(p+2), h^(p+4), ...
    cols = Vector{Vector{Float64}}(undef, nterms)
    cols[1] = ones(N)
    for k in 2:nterms
        cols[k] = h .^ (p + 2*(k - 2))
    end
    X = hcat(cols...)

    # weights
    W = Diagonal(1.0 ./ σ)

    Xw = W * X
    yw = W * y

    # --- WLS solve ---
    params = Xw \ yw

    # --- covariance (Toussaint style) ---
    # (Xᵀ W² X)^(-1)
    Cov = inv(transpose(X) * (W^2) * X)

    param_errors = sqrt.(diag(Cov))

    # diagnostics
    yhat  = X * params
    resid = y .- yhat

    chisq = sum((resid ./ σ).^2)
    dof   = length(y) - length(params)
    redchisq = chisq / dof

    return (;
        estimate = params[1],
        estimate_error = param_errors[1],
        params   = params,
        param_errors = param_errors,
        cov      = Cov,
        chisq    = chisq,
        redchisq = redchisq,
        dof      = dof
    )
end

"""
    print_fit_result(
        fit
    ) -> Nothing

Print a formatted summary of a convergence fit result.

# Function description
This routine prints each fitted parameter `λ_k` with its one-sigma uncertainty
using `AvgErrFormatter.avgerr_e2d_from_float`, followed by χ² diagnostics and the
extrapolated result `I(h→0)`.

The output formatting and ordering are intentionally kept identical to the
original implementation.

# Arguments
- `fit`: Fit result object (typically the `NamedTuple` returned by `fit_convergence`)
  that provides the fields:
  - `params`
  - `param_errors`
  - `chisq`
  - `dof`
  - `redchisq`
  - `estimate`
  - `estimate_error`

# Returns
- `nothing`.
"""
function print_fit_result(
    fit
)

    for i in eachindex(fit.params)
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.params[i], fit.param_errors[i])
        println("           λ_$(i-1) = $(tmp_str)")
    end

    println()

    @printf("Chi^2 / d.o.f. = %.12e / %d = %.12e\n",
            fit.chisq,
            fit.dof,
            fit.redchisq)

    tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.estimate, fit.estimate_error)
    println("Result (h→0)   = $(tmp_str)")

    println()
end

end  # module FitConvergence