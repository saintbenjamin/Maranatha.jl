include("experiments/integrand_F0000GammaEminus1.jl")

register_F0000_integrand!()

# ff_tilde(x)  = gtilde_F0000(x; p=4)
ff(x)  = g_F0000_pure(x)

bounds = (0.0, 1.0)
use_error_jet = false

@testset "F0000GammaEminus1 1D" begin
    @testset "Gauss 2-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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
        dim = 1
        rule = :gauss_p3
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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
        dim = 1
        rule = :gauss_p4
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 5-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p5
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 6-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p6
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 7-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p7
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 8-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p8
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 9-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p9
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 10-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p10
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 11-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p11
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "Gauss 12-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p12
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 2-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p2
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 2-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p2
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 3-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p3
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 3-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p3
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 4-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 4-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 5-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p5
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 3
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 5-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p5
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 3
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 6-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p6
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 4
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 6-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p6
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 4
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 7-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p7
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 5
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 7-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p7
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 5
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 8-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p8
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 6
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 8-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p8
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 6
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 9-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p9
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 7
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 9-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p9
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 7
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 10-point interp LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p10
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 8
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

    @testset "B-spline 10-point smooth LU_ININ" begin
        dim = 1
        rule = :bspline_smooth_p10
        boundary = :LU_ININ
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        ns .+= 8
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000"
        save_path = nothing
        write_summary = false
        save_file = false
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
            use_error_jet=use_error_jet,
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

# Registry sanity
@test :F0000 in Maranatha.Integrands.available_integrands()

# Construct preset integrand via registry
ff_here = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)
bounds = (0.0, 1.0)
use_error_jet = false

@testset "Integrand preset API (F0000)" begin

    @testset "Gauss LU_EXEX (preset)" begin
        dim = 1
        rule = :gauss_p12
        boundary = :LU_EXEX
        ns = [3, 4, 5, 6, 7, 8, 9]
        ns .+= 14
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 2
        result_string = "F0000_preset"
        save_path = nothing
        write_summary = false
        save_file = false
        run_result = run_Maranatha(
            ff_here, 
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
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
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

@testset "Preset vs raw integrand consistency (spot-check)" begin

    t = 0.37
    # f_raw = t -> gtilde_F0000(t; p=4, eps=1e-15)
    # f_pre = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)
    f_raw = t -> g_F0000_pure(t)
    f_pre = Maranatha.Integrands.integrand(:F0000)

    @test isfinite(f_raw(t))
    @test isfinite(f_pre(t))

    # Same underlying formula expected (exact match in current design)
    @test f_raw(t) == f_pre(t)
end