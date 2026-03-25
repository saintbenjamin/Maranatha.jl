@testset "1D rules" begin
    ff(x) = sin(x)
    bounds = (0.0, π)
    use_error_jet = false

    @testset "Gauss 2-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "1D"
        save_path = nothing
        write_summary = false
        save_file = false
        use_cuda = false
        real_type = Float64
        run_result = @time run_Maranatha(
            ff, 
            bounds...; 
            dim = dim, 
            nsamples = ns,
            rule = rule, 
            boundary = boundary, 
            err_method = err_method,
            fit_terms = fit_terms, 
            nerr_terms = nerr_terms,
            ff_shift = ff_shift, 
            use_error_jet = use_error_jet,
            name_prefix = result_string,
            save_path = save_path,
            write_summary = write_summary,
            use_cuda = use_cuda,
            real_type = real_type
        )
        fit_result = least_chi_square_fit(
            run_result,
            fit_func_terms = fit_terms,
            ff_shift = ff_shift,
            nerr_terms = nerr_terms
        )
        print_fit_result(fit_result)
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
            run_result,
            fit_result; 
            name = result_string,
            figs_dir = save_path,
            save_file = save_file
        )
    end

    @testset "Gauss 3-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p3
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "1D"
        save_path = nothing
        write_summary = false
        save_file = false
        use_cuda = false
        real_type = Float64
        run_result = @time run_Maranatha(
            ff, 
            bounds...; 
            dim = dim, 
            nsamples = ns,
            rule = rule, 
            boundary = boundary, 
            err_method = err_method,
            fit_terms = fit_terms, 
            nerr_terms = nerr_terms,
            ff_shift = ff_shift, 
            use_error_jet = use_error_jet,
            name_prefix = result_string,
            save_path = save_path,
            write_summary = write_summary,
            use_cuda = use_cuda,
            real_type = real_type
        )
        fit_result = least_chi_square_fit(
            run_result,
            fit_func_terms = fit_terms,
            ff_shift = ff_shift,
            nerr_terms = nerr_terms
        )
        print_fit_result(fit_result)
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
            run_result,
            fit_result; 
            name = result_string,
            figs_dir = save_path,
            save_file = save_file
        )
    end

    @testset "Gauss 4-point LU_EXEX" begin
        dim = 1
        rule = :gauss_p4
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "1D"
        save_path = nothing
        write_summary = false
        save_file = false
        use_cuda = false
        real_type = Float64
        run_result = @time run_Maranatha(
            ff, 
            bounds...; 
            dim = dim, 
            nsamples = ns,
            rule = rule, 
            boundary = boundary, 
            err_method = err_method,
            fit_terms = fit_terms, 
            nerr_terms = nerr_terms,
            ff_shift = ff_shift, 
            use_error_jet = use_error_jet,
            name_prefix = result_string,
            save_path = save_path,
            write_summary = write_summary,
            use_cuda = use_cuda,
            real_type = real_type
        )
        fit_result = least_chi_square_fit(
            run_result,
            fit_func_terms = fit_terms,
            ff_shift = ff_shift,
            nerr_terms = nerr_terms
        )
        print_fit_result(fit_result)
        assert_result_sane(run_result)
        @test all(isfinite, run_result.avg) &&
              all(e -> isfinite(getproperty(e, hasproperty(e, :total) ? :total : :estimate)),
                  run_result.err)
        DO_PLOT && plot_convergence_result(
            run_result,
            fit_result; 
            name = result_string,
            figs_dir = save_path,
            save_file = save_file
        )
    end
end