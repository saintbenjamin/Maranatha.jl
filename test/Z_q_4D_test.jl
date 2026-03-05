@testset "Z_q 4D" begin
    announce("Z_q 4D")

    ff(x1,x2,x3,x4) = Maranatha.Z_q.integrand_Z_q((x1,x2,x3,x4))
    bounds=(0.0,π)
    use_threads = false

    @testset "Gauss 2-point LU_EXEX ForwardDiff.jl" begin
        announce("4D rules :: Gauss 2-point LU_EXEX ForwardDiff.jl")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 4
        result_string = "Z_q_ForwardDiff"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "Gauss 2-point LU_EXEX FastDifferentiation.jl" begin
    #     announce("4D rules :: Gauss 2-point LU_EXEX FastDifferentiation.jl")
    #     dim = 4
    #     rule = :gauss_p2
    #     boundary = :LU_EXEX
    #     ns = [2, 3, 4, 5, 6, 7, 8, 9]
    #     ns .+= 20
    #     err_method = :fastdifferentiation # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
    #     nerr_terms = 1
    #     ff_shift = 0
    #     fit_terms = 4
    #     result_string = "Z_q_FastDiff"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "Gauss 2-point LU_EXEX TaylorSeries.jl" begin
    #     announce("4D rules :: Gauss 2-point LU_EXEX TaylorSeries.jl")
    #     dim = 4
    #     rule = :gauss_p2
    #     boundary = :LU_EXEX
    #     ns = [2, 3, 4, 5, 6, 7, 8, 9]
    #     ns .+= 20
    #     err_method = :taylorseries # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
    #     nerr_terms = 1
    #     ff_shift = 0
    #     fit_terms = 4
    #     result_string = "Z_q_Taylor"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end
end