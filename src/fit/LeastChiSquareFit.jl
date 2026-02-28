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
using ..JobLoggerTools

export fit_convergence, print_fit_result

"""
    fit_convergence(
        hs,
        estimates,
        errors,
        rule::Symbol;
        nterms::Int = 2
    )

Perform a ``h \\to 0`` (zero step-size limit) extrapolation using least-``\\chi^2`` fitting for the quadrature estimate, and return both best-fit parameters and
their uncertainties.

# Function description
This routine fits a rule-dependent convergence ansatz that is **linear in its
parameters**, using weighted least squares (WLS). The step size is provided by
the caller as ``\\displaystyle{h = \\frac{b-a}{N}}`` (via `hs`), and the leading convergence power ``p``
is inferred from the chosen quadrature `rule`.

The convergence model is selected by the number of basis terms `nterms`
(including the constant term). The design matrix is constructed as:

- column 1: ``h^0`` (constant term)
- columns 2..nterms: ``h^p``, ``h^{(p+2)}``, ``h^{(p+4)}``, ...

Equivalently, the fitted form is:
```math
I(h) = I_0 + C_1 \\, h^p + C_2 \\, h^{p+2} + C_3 \\, h^{p+4} + \\ldots
```

## Solve (WLS)
Let ``X`` be the design matrix, ``y`` the estimates, and ``\\sigma`` the pointwise errors.
Weights are constructed as ``W = \\sigma_{ii}^{-1}``, and the WLS normal equations
are solved in the numerically stable form:
```math
\\left( W \\,X \\right) \\; \\bm{\\lambda} = \\left( W \\, y \\right) \\,,
```
yielding ``\\displaystyle{\\bm{\\lambda} =
\\begin{bmatrix}
I_0 & C_1 & C_2 & \\cdots
\\end{bmatrix}^{\\mathsf{T}}}``.

## Parameter covariance and errors
This implementation computes Hessian-based propagation, but evaluates the needed
inverse-Hessian actions via a Cholesky factorization rather than forming `inv(H)`
explicitly.

1) Build the normal matrix ``A = X^{\\mathsf{T}} \\; W^2 \\; X``.
2) Define the χ² Hessian as ``H = 2\\,A``.
3) Factorize the Hessian ``H = L \\, L^{\\mathsf{T}}`` via `cholesky(Symmetric(H))` (requires *Symmetric Positive Definite (SPD)*).
4) Define ``H^{-1}`` implicitly through linear solves with the Cholesky factor.
5) The parameter covariance matrix ``V`` is then taken as
```math
V = \\Delta = 4 \\, H^{-1} \\, A \\, H^{-1} \\,,
```
and the ``1 \\, \\sigma`` errors of fitting parameters are ``\\sqrt{\\diag(V)}``.

### Why ``H = 2 \\, A``?
For the WLS objective ``\\chi^{2}(\\bm{\\lambda}) = \\left\\lvert W \\left( X \\, \\bm{\\lambda} - \\mathbf{y} \\right) \\right\\rvert^2``,
the gradient is ``\\nabla \\chi^2 = 2 X^{\\mathsf{T}} \\, W^2 \\, \\left( X \\, \\bm{\\lambda} - \\mathbf{y} \\right)`` and the Hessian is ``\\nabla^2 \\chi^2 = 2 X^{\\mathsf{T}} \\, W^2 \\, X = 2 \\, A``. So this factor of 2 is exact for this model.

### Notes:
- If you prefer the standard WLS covariance (Gauss-Markov for linear models),
  you can instead use ``V = \\left( X^{\\mathsf{T}} \\, W^2 \\, X \\right)^{-1}``.
- The returned `cov` is intended for downstream use (e.g. to draw a fit-band
  via ``\\sigma_{\\text{fit}}^2(h) = \\varphi(h)^{\\mathsf{T}} \\, V \\, \\varphi(h)``).

# Arguments
- `hs`: Vector-like collection of step sizes ``\\displaystyle{h = \\frac{b-a}{N}}``.
- `estimates`: Vector-like collection of quadrature estimates ``I(h)``.
- `errors`: Vector-like collection of error estimates associated with ``I(h)``.
- `rule`: Quadrature rule symbol used to select the leading convergence power `p`.

# Keyword arguments
- `nterms::Int=2`: Number of basis terms in the convergence model (including the
  constant term). Must satisfy `nterms```\\ge 2``.

# Returns
A `NamedTuple` with the following fields:
- `estimate::Float64`: Extrapolated value ``I_0 = I(h \\to 0)`` (i.e. `params[1]`).
- `estimate_error::Float64`: One-sigma uncertainty for ``I_0``, taken from the
  covariance diagonal.
- `params::Vector{Float64}`: Fitted parameter vector `[I0, C1, C2, ...]`.
- `param_errors::Vector{Float64}`: ``1 \\, \\sigma`` uncertainties for `params`.
- `cov::Matrix{Float64}`: Parameter covariance matrix.
- `chisq::Float64`: ``\\chi^2`` value.
- `redchisq::Float64`: ``\\chi^2/\\text{d.o.f.}``.
- `dof::Int`: Degrees of freedom, `length(y) - length(params)`.

# Errors
- Throws an error if `rule` is not recognized.
- Throws an error if `nterms < 2`.
- Throws an error if the Hessian is not positive definite (Cholesky fails).
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
        JobLoggerTools.error_benji("Unknown rule")

    h = collect(float.(hs))
    y = collect(float.(estimates))
    σ = abs.(float.(errors))

    N = length(h)

    if nterms < 2
        JobLoggerTools.error_benji("nterms must be >= 2 (got $nterms)")
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

    # # --- covariance (Method 1) ---
    # # (Xᵀ W² X)^(-1)
    # # A = transpose(X) * (W^2) * X
    # # Cov = inv(A)
    # Cov = inv(transpose(X) * (W^2) * X)

    # param_errors = sqrt.(diag(Cov))

    # --- covariance (Method 2) ---
    # Build A = Xᵀ W² X
    A = transpose(X) * (W^2) * X

    # # Hessian = 2 * A
    # Hess = 2.0 .* A
    # inv_Hess = inv(Hess)

    # # Delta = 4 * inv_Hess * A * inv_Hess
    # Delta = 4.0 .* inv_Hess * A * inv_Hess

    # Cov = Delta
    # param_errors = sqrt.(diag(Delta))

    Hess = 2.0 .* A
    F = cholesky(Symmetric(Hess))          # Hess must be SPD

    # Cov = 4 * inv(Hess) * A * inv(Hess)  (computed via solves)
    M   = F \ A                            # M = inv(Hess) * A
    Cov = 4.0 .* ((F \ transpose(M))')     # Cov = 4 * M * inv(Hess)

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
    jobid = nothing

    for i in eachindex(fit.params)
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.params[i], fit.param_errors[i])
        JobLoggerTools.println_benji("           λ_$(i-1) = $(tmp_str)", jobid)
    end

    JobLoggerTools.println_benji("",jobid)

    JobLoggerTools.println_benji(
        @sprintf(
            "Chi^2 / d.o.f. = %.12e / %d = %.12e",
            fit.chisq,
            fit.dof,
            fit.redchisq
        ), jobid
    )

    tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.estimate, fit.estimate_error)
    JobLoggerTools.println_benji("Result (h→0)   = $(tmp_str)", jobid)

    JobLoggerTools.println_benji("",jobid)
end

end  # module FitConvergence