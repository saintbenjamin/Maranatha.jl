@testset "3D rules" begin
    announce("3D rules")

    ff(x, y, z) = exp(-x^2 - y^2 - z^2)
    bounds = (0.0, 1.0)
    use_threads = false


    @testset "3D B-spline 2-point interp LU_ININ" begin
        announce("3D rules :: B-spline 2-point interp LU_ININ")
        dim = 3
        rule = :bspline_interp_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 2-point smooth LU_ININ" begin
        announce("3D rules :: B-spline 2-point smooth LU_ININ")
        dim = 3
        rule = :bspline_smooth_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 3-point LU_ININ" begin
        announce("3D rules :: B-spline 3-point LU_ININ")
        dim = 3
        rule = :bspline_interp_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 4-point LU_ININ" begin
        announce("3D rules :: B-spline 4-point LU_ININ")
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 5-point LU_ININ" begin
        announce("3D rules :: B-spline 5-point LU_ININ")
        dim = 3
        rule = :bspline_interp_p5
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 6-point LU_ININ" begin
        announce("3D rules :: B-spline 6-point LU_ININ")
        dim = 3
        rule = :bspline_interp_p6
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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

    @testset "3D B-spline 7-point LU_ININ" begin
        announce("3D rules :: B-spline 7-point LU_ININ")
        dim = 3
        rule = :bspline_interp_p7
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 10
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "3D"
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
end