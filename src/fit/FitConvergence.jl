module FitConvergence

using LinearAlgebra
using Statistics
using Printf
using ..AvgErrFormatter

export fit_convergence, print_fit_result

"""
    fit_convergence(hs, estimates, errors, rule::Symbol; dim::Int=1)

Perform weighted linear least-χ² extrapolation for I(h → 0).

The model is linear in parameters:

    I(h) = I0 + C1*h^p + C2*h^(p+2)     (1D case)

This allows a stable weighted linear least squares solve
instead of nonlinear curve_fit.

Returns:
    NamedTuple(
        estimate = extrapolated I0,
        params   = fitted parameter vector,
        chisq    = chi-square,
        redchisq = reduced chi-square,
        dof      = degrees of freedom
    )
"""
function fit_convergence(hs, estimates, errors, rule::Symbol; dim::Int=1)

    p =
        rule == :simpson13_close            ? 4 :
        rule == :simpson13_open  ? 4 :
        rule == :simpson38_close            ? 4 :
        rule == :simpson38_open  ? 4 :
        rule == :bode_close                 ? 6 :
        rule == :bode_open       ? 6 :
        error("Unknown rule")

    h = collect(float.(hs))
    y = collect(float.(estimates))
    σ = map(e -> e > 0 ? float(e) : 1e-8, errors)

    if dim == 1
        X = hcat(ones(length(h)), h.^p, h.^(p+2), h.^(p+4))
    else
        X = hcat(ones(length(h)), h.^p)
    end

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
    print_fit_result(fit)

Pretty formatted output similar to legacy least-χ² C routine.
"""
function print_fit_result(fit)

    # println("\nFitting parameters (float)")
    # println("--------------------")

    # for i in eachindex(fit.params)
    #     @printf("λ_%-2d = % .12e (± %.12e)\n",
    #             i-1,
    #             fit.params[i],
    #             fit.param_errors[i])
    # end
    
    # println()

    for i in eachindex(fit.params)
        tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.params[i], fit.param_errors[i])
        println("           λ_$(i-1) = $(tmp_str)")
    end
    
    println()

    @printf("Chi^2 / d.o.f. = %.12e / %d = %.12e\n",
            fit.chisq,
            fit.dof,
            fit.redchisq)
    # @printf("Result (h→0)   = %.12e\n", fit.estimate)
    # @printf("Error  (h→0)   = %.12e\n", fit.estimate_error)
    tmp_str = AvgErrFormatter.avgerr_e2d_from_float(fit.estimate, fit.estimate_error)
    println("Result (h→0)   = $(tmp_str)")

    println()
end

end # module