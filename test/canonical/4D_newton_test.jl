@testset "4D rules" begin
    ff(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
    bounds = (0.0, 1.0)
    use_error_jet = false

    @testset "Trapezoidal LU_ININ" begin
        dim = 4
        rule = :newton_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Trapezoidal LU_INEX" begin
        dim = 4
        rule = :newton_p2
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Trapezoidal LU_EXIN" begin
        dim = 4
        rule = :newton_p2
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Trapezoidal LU_EXEX" begin
        dim = 4
        rule = :newton_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 1/3 LU_ININ" begin
        dim = 4
        rule = :newton_p3
        boundary = :LU_ININ
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 1/3 LU_INEX" begin
        dim = 4
        rule = :newton_p3
        boundary = :LU_INEX
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 1/3 LU_EXIN" begin
        dim = 4
        rule = :newton_p3
        boundary = :LU_EXIN
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 1/3 LU_EXEX" begin
        dim = 4
        rule = :newton_p3
        boundary = :LU_EXEX
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 3/8 LU_ININ" begin
        dim = 4
        rule = :newton_p4
        boundary = :LU_ININ
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 3/8 LU_INEX" begin
        dim = 4
        rule = :newton_p4
        boundary = :LU_INEX
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 3/8 LU_EXIN" begin
        dim = 4
        rule = :newton_p4
        boundary = :LU_EXIN
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Simpson 3/8 LU_EXEX" begin
        dim = 4
        rule = :newton_p4
        boundary = :LU_EXEX
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Bode LU_ININ" begin
        dim = 4
        rule = :newton_p5
        boundary = :LU_ININ
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 0
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Bode LU_INEX" begin
        dim = 4
        rule = :newton_p5
        boundary = :LU_INEX
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Bode LU_EXIN" begin
        dim = 4
        rule = :newton_p5
        boundary = :LU_EXIN
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "Bode LU_EXEX" begin
        dim = 4
        rule = :newton_p5
        boundary = :LU_EXEX
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 2
        err_method = :refinement # :forwarddiff , :taylorseries , :enzyme , 
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
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

    @testset "6-point LU_ININ" begin
        dim = 4
        rule = :newton_p6
        boundary = :LU_ININ
        ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
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

    @testset "6-point LU_INEX" begin
        dim = 4
        rule = :newton_p6
        boundary = :LU_INEX
        ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
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

    @testset "6-point LU_EXIN" begin
        dim = 4
        rule = :newton_p6
        boundary = :LU_EXIN
        ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
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

    @testset "6-point LU_EXEX" begin
        dim = 4
        rule = :newton_p6
        boundary = :LU_EXEX
        ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
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
end