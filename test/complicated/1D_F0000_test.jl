include("experiments/integrand_F0000GammaEminus1.jl")

register_F0000_integrand!()

ff_tilde(x)  = gtilde_F0000(x; p=4)
ff(x)  = g_F0000_raw(x)

bounds = (0.0, 1.0)
use_threads = false

@testset "F0000GammaEminus1 1D" begin
    @testset "Gauss LU_EXEX FastDifferentiation.jl" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :fastdifferentiation # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 2
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000_FastDiff"
        save_path = "."
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

    @testset "Gauss LU_EXEX ForwardDiff.jl" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000_FwrdDiff"
        save_path = "."
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

    @testset "Gauss LU_EXEX TaylorSeries.jl" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :taylorseries # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000_Taylor"
        save_path = "."
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

    @testset "B-spline 3-point LU_ININ" begin
        dim = 1
        rule = :bspline_interp_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :fastdifferentiation # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000"
        save_path = "."
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
use_threads = false

@testset "Integrand preset API (F0000)" begin

    @testset "Gauss LU_EXEX (preset)" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000_preset"
        save_path = "."
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
    f_raw = t -> gtilde_F0000(t; p=4, eps=1e-15)
    f_pre = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)

    @test isfinite(f_raw(t))
    @test isfinite(f_pre(t))

    # Same underlying formula expected (exact match in current design)
    @test f_raw(t) == f_pre(t)
end