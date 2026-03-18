using Maranatha

include("../scripts/experiments/integrand_Z_q.jl")

ff(x1,x2,x3,x4) = integrand_Z_q((x1,x2,x3,x4))

# bounds=(0.0,π)
bounds=(0.0, 3.141592653589793)
use_threads = true
dim = 4
rule = :gauss_p4
boundary = :LU_EXEX
err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
nerr_terms = 3
ff_shift = 0
fit_terms = 4
result_string = "Z_q"
save_path = joinpath("../samples", "jld2")
write_summary = true
save_file=true
use_cuda=false
use_error_jet=false

ns = [3,4,5,6,7]
ns .+= 10

Maranatha.Utils.JobLoggerTools.log_stage_benji("BEGIN run_Maranatha")

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
    use_error_jet=use_error_jet,
    name_prefix=result_string,
    save_path=save_path,
    write_summary=write_summary,
    use_cuda=use_cuda,
)

Maranatha.Utils.JobLoggerTools.log_stage_benji("END run_Maranatha")