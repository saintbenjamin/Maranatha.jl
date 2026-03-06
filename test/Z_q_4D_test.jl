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
        save_file=true
        run_result = Maranatha.Runner.run_Maranatha(
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
            use_threads=use_threads
        )
        fit_result = Maranatha.LeastChiSquareFit.least_chi_square_fit(
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
        Maranatha.LeastChiSquareFit.print_fit_result(fit_result)
        assert_result_sane(run_result); @test all(isfinite, run_result.avg) && all(e -> isfinite(e.total), run_result.err)
        maybe_plot(
            bounds..., 
            result_string,
            run_result.h, 
            run_result.avg, 
            run_result.err, fit_result;
            rule=rule, 
            boundary=boundary,
            save_file=save_file
        )
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
    #     run_result = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=err_method,
    #         fit_terms=fit_terms, nerr_terms=nerr_terms,
    #         ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     fit_result = Maranatha.LeastChiSquareFit.least_chi_square_fit(
    #         run_result.a,
    #         run_result.b,
    #         run_result.h,
    #         run_result.avg,
    #         run_result.err,
    #         run_result.rule,
    #         run_result.boundary;
    #         nterms=fit_terms,
    #         ff_shift=ff_shift,
    #         nerr_terms=nerr_terms
    #     )
    #     assert_result_sane(run_result); @test all(isfinite, run_result.avg) && all(e -> isfinite(e.total), run_result.err)
    #     maybe_plot(
    #         bounds..., result_string,
    #         run_result.h, run_result.avg, run_result.err, fit_result;
    #         rule=rule, boundary=boundary
    #     )
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
    #     run_result = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=err_method,
    #         fit_terms=fit_terms, nerr_terms=nerr_terms,
    #         ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     fit_result = Maranatha.LeastChiSquareFit.least_chi_square_fit(
    #         run_result.a,
    #         run_result.b,
    #         run_result.h,
    #         run_result.avg,
    #         run_result.err,
    #         run_result.rule,
    #         run_result.boundary;
    #         nterms=fit_terms,
    #         ff_shift=ff_shift,
    #         nerr_terms=nerr_terms
    #     )
    #     assert_result_sane(run_result); @test all(isfinite, run_result.avg) && all(e -> isfinite(e.total), run_result.err)
    #     maybe_plot(
    #         bounds..., result_string,
    #         run_result.h, run_result.avg, run_result.err, fit_result;
    #         rule=rule, boundary=boundary
    #     )
    # end
end