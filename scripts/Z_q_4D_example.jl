using Maranatha

include("experiments/integrand_Z_q.jl")

ff(x1,x2,x3,x4) = integrand_Z_q((x1,x2,x3,x4))
bounds=(0.0,π)
use_error_jet = false

dim = 4
rule = :gauss_p2
boundary = :LU_EXEX
ns = [2, 3, 4, 5, 6, 7, 8, 9]
ns .+= 20
err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
nerr_terms = 1
ff_shift = 0
fit_terms = 4
result_string = "Z_q_ForwardDiff"
save_file=true

@time run_result = run_Maranatha(
    ff, 
    bounds...; 
    dim=dim, 
    nsamples=ns,
    rule=rule, 
    boundary=boundary, 
    err_method=err_method,
    fit_terms=fit_terms, 
    nerr_terms=nerr_terms,
    ff_shift=ff_shift, 
    use_error_jet=use_error_jet
)

@time fit_result = least_chi_square_fit(
    run_result.a,
    run_result.b,
    run_result.h,
    run_result.avg,
    run_result.err,
    run_result.rule,
    run_result.boundary;
    nterms=fit_terms,
    ff_shift=ff_shift,
    nerr_terms=nerr_terms
)

@time print_fit_result(fit_result)

@time plot_convergence_result(
    bounds..., 
    result_string,
    run_result.h, 
    run_result.avg, 
    run_result.err, fit_result;
    rule=rule, 
    boundary=boundary,
    save_file=save_file
)