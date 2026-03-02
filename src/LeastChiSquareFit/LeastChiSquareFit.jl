# ============================================================================
# src/generator/LeastChiSquareFit.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module LeastChiSquareFit

using LinearAlgebra
using Statistics
using Printf
using ..AvgErrFormatter
using ..JobLoggerTools
using ..ErrorEstimate

export least_chi_square_fit, print_fit_result

"""
    least_chi_square_fit(
        a::Real,
        b::Real,
        hs,
        estimates,
        errors,
        rule::Symbol,
        boundary::Symbol;
        nterms::Int = 2,
        ff_shift::Int = 0
    )

Perform a ``h \\to 0`` (zero step-size limit) extrapolation using least-``\\chi^2`` fitting
for the quadrature estimate, and return both best-fit parameters and their uncertainties.

# Function description
This routine fits a rule- and boundary-dependent convergence ansatz that is **linear in its
parameters**, using weighted least squares (WLS). The step size is provided by the caller as
``\\displaystyle{h = \\frac{b-a}{N}}`` (via `hs`).

The convergence exponents are inferred from the composite Newton-Cotes midpoint residual model,
which depends on `(rule, boundary)` and a representative subdivision count `Nref` derived from
the smallest step size in `hs`.

## Exponent selection (midpoint residual model + forward shift)
First, a list of candidate residual powers is obtained via
[`Maranatha.ErrorEstimate._leading_residual_ks_with_center`](@ref):
- `powers_all = ks`.

These are then **sliced** to build the fit basis using the optional forward-shift `ff_shift`:

- The model uses exactly `need = nterms - 1` non-constant powers.
- The selected powers are:
```math
\\texttt{powers} = \\texttt{powers\\_all[ (1+ff\\_shift) : (1+ff\\_shift+need-1) ]}
```
This is useful when the integrand makes the true leading-order coefficient vanish
(e.g., the corresponding midpoint derivative is identically zero), so the fit may benefit
from skipping the nominal leading power and using the next ones.

## Convergence model and design matrix

The fitted form is:

```math
I(h) = I_0 + C_1 \\, h^{p_1} + C_2 \\, h^{p_2} + \\cdots + C_{\\texttt{need}} \\, h^{p_{\\texttt{need}}}
```

with exponent list ``\\texttt{powers} = [p_1, p_2, ..., p_\\texttt{need}]`` selected as described above.

The design matrix is constructed as:

* column ``1``  : ``h^0`` (constant term)
* column ``t+1``: ``h^{\\texttt{powers[t]}}`` for ``t = 1,\\ldots,\\texttt{need}``

## Solve (WLS)

Let ``X`` be the design matrix, ``y`` the estimates, and ``\\sigma`` the pointwise errors.
Weights are constructed as ``\\displaystyle{W = \\mathrm{diag}\\left(\\frac{1}{\\sigma}\\right)}``, and the WLS problem is solved as:
```math
\\left( W \\, X \\right) \\bm{\\lambda} = \\left( W \\, y \\right),
```
yielding ``\\bm{\\lambda} = [I_0, C_1, C_2, \\ldots]^{\\mathsf{T}}``.

## Parameter covariance and errors

This implementation computes Hessian-based propagation, using a Cholesky factorization rather
than forming `inv(H)` explicitly.

1. Build the normal matrix ``A = X^{\\mathsf{T}} W^2 X``.
2. Define the ``\\chi^2`` Hessian as ``H = 2A``.
3. Factorize ``H`` via `cholesky(Symmetric(H))` (requires SPD (symmetric positive definite)).
4. Compute the covariance as:

```math
V = 4 \\, H^{-1} \\, A \\, H^{-1}.
```

The ``1\\,\\sigma`` parameter errors are ``\\sqrt{\\mathrm{diag}(V)}``.

# Arguments

* `a`, `b`:
  Integration bounds (used only to derive `Nref` from the smallest `h` in `hs`).
* `hs`:
  Vector-like collection of step sizes ``h``.
* `estimates`:
  Vector-like collection of quadrature estimates ``I(h)``.
* `errors`:
  Vector-like collection of error estimates for ``I(h)`` (absolute values are used internally).
* `rule`:
  Quadrature rule symbol used by the midpoint residual model.
* `boundary`:
  Boundary pattern symbol (`:LCRC`, `:LORC`, `:LCRO`, `:LORO`) used by the midpoint residual model.

# Keyword arguments

* `nterms::Int = 2`:
  Number of basis terms in the convergence model (including the constant term).
  Must satisfy `nterms```\\ge 2``.
* `ff_shift::Int = 0`:
  Forward shift applied to the candidate residual-power list before selecting `nterms-1` powers.
  Must satisfy `ff_shift```\\ge 0``. If the shift makes the requested slice impossible, an error is thrown.

# Returns

A `NamedTuple` with the following fields:

* `estimate::Float64`:
  Extrapolated value ``I_0 = I(h \\to 0)`` (i.e. `params[1]`).
* `error_estimate::Float64`:
  One-sigma uncertainty for ``I_0`` from the covariance diagonal.
* `params::Vector{Float64}`:
  Fitted parameter vector ``[I_0, C_1, C_2, \\ldots]``.
* `param_errors::Vector{Float64}`:
  ``1\\sigma`` uncertainties for `params`.
* `cov::Matrix{Float64}`:
  Parameter covariance matrix.
* `powers::Vector{Int}`:
  Exponent vector used by the fit basis, returned as `vcat(0, powers)` to align with `params`.
* `chisq::Float64`:
  ``\\chi^2`` value.
* `redchisq::Float64`:
  ``\\chi^2/\\text{d.o.f.}`` value.
* `dof::Int`:
  Degrees of freedom, `length(y) - length(params)`.

# Errors

* Throws an error if `nterms < 2`.
* Throws an error if `ff_shift < 0`.
* Throws an error if there are not enough residual powers available after applying `ff_shift`.
* Throws an error if the Hessian is not positive definite (Cholesky decomposition fails).
* Note: If `dof == 0`, `redchisq = chisq / dof` follows IEEE rules and may become `Inf`/`NaN`.
"""
function least_chi_square_fit(
    a::Real,
    b::Real,
    hs, 
    estimates, 
    errors, 
    rule::Symbol,
    boundary::Symbol; 
    nterms::Int=2,
    ff_shift::Int=0
)
    # ------------------------------------------------------------
    # Determine leading convergence power automatically
    # using composite NC residual model (midpoint expansion)
    # ------------------------------------------------------------

    Nref = round(Int, (b - a) / minimum(float.(hs)))

    ks, _center = ErrorEstimate._leading_residual_ks_with_center(
        rule, boundary, Nref; nterms=nterms, kmax=256
    )

    # ------------------------------------------------------------
    # Select fit powers with optional forward-shift
    # ------------------------------------------------------------

    powers_all = ks

    (nterms >= 2)   || JobLoggerTools.error_benji("nterms must be >= 2 (got $nterms)")
    (ff_shift >= 0) || JobLoggerTools.error_benji("ff_shift must be ≥ 0 (got $ff_shift)")

    need  = nterms - 1
    start = 1 + ff_shift
    stop  = start + need - 1

    (stop <= length(powers_all)) || JobLoggerTools.error_benji(
        "Not enough residual powers: need $(need) terms after ff_shift=$ff_shift, but only $(length(powers_all)) available."
    )

    powers = powers_all[start:stop]

    JobLoggerTools.println_benji(
        "fit powers (h^p), ff_shift=$(ff_shift): " * join(string.(powers), ", ")
    )

    h = collect(float.(hs))
    y = collect(float.(estimates))
    σ = abs.(float.(errors))

    N = length(h)

    # Design matrix:
    #   col 1: 1
    #   col t+1: h^(powers[t])  (t = 1..nterms-1)
    cols = Vector{Vector{Float64}}(undef, nterms)
    cols[1] = ones(N)

    for t in 1:need
        cols[t+1] = h .^ powers[t]
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
        error_estimate = param_errors[1],
        params   = params,
        param_errors = param_errors,
        cov      = Cov,
        powers   = vcat(0, powers), 
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
This routine prints each fitted parameter ``\\lambda_k`` with its ``1 \\, \\sigma`` uncertainty
using [`Maranatha.AvgErrFormatter.avgerr_e2d_from_float`](@ref), followed by ``\\chi^2`` diagnostics 
and the extrapolated result ``I(h \\to 0)``.

The output formatting and ordering are intentionally kept identical to the
original implementation.

# Arguments
- `fit`: Fit result object (typically the `NamedTuple` returned by [`least_chi_square_fit`](@ref))
  that provides the fields:
    - `estimate::Float64`: Extrapolated value ``I_0 = I(h \\to 0)`` (i.e. `params[1]`).
    - `error_estimate::Float64`: One-sigma uncertainty for ``I_0``, taken from the
    covariance diagonal.
    - `params::Vector{Float64}`: Fitted parameter vector `[I0, C1, C2, ...]`.
    - `param_errors::Vector{Float64}`: ``1 \\, \\sigma`` uncertainties for `params`.
    - `cov::Matrix{Float64}`: Parameter covariance matrix.
    - `chisq::Float64`: ``\\chi^2`` value.
    - `redchisq::Float64`: ``\\chi^2/\\text{d.o.f.}``.
    - `dof::Int`: Degrees of freedom, `length(y) - length(params)`.

# Returns
- `nothing`.
"""
function print_fit_result(
    fit
)
    jobid = nothing

    for i in eachindex(fit.params)
        if !isfinite(fit.params[i]) || !isfinite(fit.param_errors[i])
            tmp_str = @sprintf("%.12e (%.12e)", fit.params[i], fit.param_errors[i])
        else
            tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.params[i], fit.param_errors[i])
        end
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

    if !isfinite(fit.estimate) || !isfinite(fit.error_estimate)
        tmp_str = @sprintf("%.12e (%.12e)", fit.estimate, fit.error_estimate)
    else
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.estimate, fit.error_estimate)
    end
    JobLoggerTools.println_benji("Result (h→0)   = $(tmp_str)", jobid)

    JobLoggerTools.println_benji("",jobid)
end

end  # module LeastChiSquareFit