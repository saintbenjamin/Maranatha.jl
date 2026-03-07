@testset "4D rules" begin
    announce("4D rules")

    ff(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Gauss 2-point LU_EXEX" begin
        announce("4D rules :: Gauss 2-point LU_EXEX")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        save_path = "."
        write_summary = true
        save_file = true
        run_result = run_Maranatha(
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
            use_threads=use_threads,
            name_prefix=result_string,
            save_path=save_path,
            write_summary=write_summary  
        )
        fit_result = least_chi_square_fit(
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
        print_fit_result(fit_result)
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

    @testset "Gauss 3-point LU_EXEX" begin
        announce("4D rules :: Gauss 3-point LU_EXEX")
        dim = 4
        rule = :gauss_p3
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        save_path = "."
        write_summary = true
        save_file = true
        run_result = run_Maranatha(
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
            use_threads=use_threads,
            name_prefix=result_string,
            save_path=save_path,
            write_summary=write_summary  
        )
        fit_result = least_chi_square_fit(
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
        print_fit_result(fit_result)
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

    @testset "Gauss 4-point LU_EXEX" begin
        announce("4D rules :: Gauss 4-point LU_EXEX")
        dim = 4
        rule = :gauss_p4
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        save_path = "."
        write_summary = true
        save_file = true
        run_result = run_Maranatha(
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
            use_threads=use_threads,
            name_prefix=result_string,
            save_path=save_path,
            write_summary=write_summary  
        )
        fit_result = least_chi_square_fit(
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
        print_fit_result(fit_result)
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
end