@testset "3D rules" begin
    ff(x, y, z) = exp(-x^2 - y^2 - z^2)
    bounds = (0.0, 1.0)
    use_error_jet = false

    @testset "B-spline 2-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p2
        boundary = :LU_ININ
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

    @testset "B-spline 2-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p2
        boundary = :LU_ININ
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

    @testset "B-spline 3-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 1
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

    @testset "B-spline 3-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 1
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

    @testset "B-spline 4-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 2
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

    @testset "B-spline 4-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 2
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

    @testset "B-spline 5-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 3
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

    @testset "B-spline 5-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 3
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

    @testset "B-spline 6-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 4
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

    @testset "B-spline 6-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 4
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

    @testset "B-spline 7-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 5
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

    @testset "B-spline 7-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 5
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

    @testset "B-spline 8-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 6
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

    @testset "B-spline 8-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 6
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

    @testset "B-spline 9-point interp LU_ININ" begin
        dim = 3
        rule = :bspline_interp_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 7
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

    @testset "B-spline 9-point smooth LU_ININ" begin
        dim = 3
        rule = :bspline_smooth_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 7
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