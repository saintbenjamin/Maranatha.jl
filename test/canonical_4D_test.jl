@testset "4D rules" begin
    announce("4D rules")

    f4D(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
    bounds = (0.0, 1.0)
    use_threads = true

    @testset "Trapezoidal LCRC" begin
        announce("4D rules :: Trapezoidal LCRC")
        dim = 4
        rule = :ns_p2
        boundary = :LCRC
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 0
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LCRO" begin
        announce("4D rules :: Trapezoidal LCRO")
        dim = 4
        rule = :ns_p2
        boundary = :LCRO
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LORC" begin
        announce("4D rules :: Trapezoidal LORC")
        dim = 4
        rule = :ns_p2
        boundary = :LORC
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LORO" begin
        announce("4D rules :: Trapezoidal LORO")
        dim = 4
        rule = :ns_p2
        boundary = :LORO
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 2
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LCRC" begin
        announce("4D rules :: Simpson 1/3 LCRC")
        dim = 4
        rule = :ns_p3
        boundary = :LCRC
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 0
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LCRO" begin
        announce("4D rules :: Simpson 1/3 LCRO")
        dim = 4
        rule = :ns_p3
        boundary = :LCRO
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 1
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LORC" begin
        announce("4D rules :: Simpson 1/3 LORC")
        dim = 4
        rule = :ns_p3
        boundary = :LORC
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 1
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LORO" begin
        announce("4D rules :: Simpson 1/3 LORO")
        dim = 4
        rule = :ns_p3
        boundary = :LORO
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 2
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LCRC" begin
        announce("4D rules :: Simpson 3/8 LCRC")
        dim = 4
        rule = :ns_p4
        boundary = :LCRC
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 0
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LCRO" begin
        announce("4D rules :: Simpson 3/8 LCRO")
        dim = 4
        rule = :ns_p4
        boundary = :LCRO
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LORC" begin
        announce("4D rules :: Simpson 3/8 LORC")
        dim = 4
        rule = :ns_p4
        boundary = :LORC
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LORO" begin
        announce("4D rules :: Simpson 3/8 LORO")
        dim = 4
        rule = :ns_p4
        boundary = :LORO
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 2
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LCRC" begin
        announce("4D rules :: Bode LCRC")
        dim = 4
        rule = :ns_p5
        boundary = :LCRC
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 0
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LCRO" begin
        announce("4D rules :: Bode LCRO")
        dim = 4
        rule = :ns_p5
        boundary = :LCRO
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LORC" begin
        announce("4D rules :: Bode LORC")
        dim = 4
        rule = :ns_p5
        boundary = :LORC
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LORO" begin
        announce("4D rules :: Bode LORO")
        dim = 4
        rule = :ns_p5
        boundary = :LORO
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 2
        result_string = "4D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f4D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "6-point LCRC" begin
    #     announce("4D rules :: 6-point LCRC")
    #     dim = 4
    #     rule = :ns_p6
    #     boundary = :LCRC
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 0
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LCRO" begin
    #     announce("4D rules :: 6-point LCRO")
    #     dim = 4
    #     rule = :ns_p6
    #     boundary = :LCRO
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 1
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LORC" begin
    #     announce("4D rules :: 6-point LORC")
    #     dim = 4
    #     rule = :ns_p6
    #     boundary = :LORC
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 1
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LORO" begin
    #     announce("4D rules :: 6-point LORO")
    #     dim = 4
    #     rule = :ns_p6
    #     boundary = :LORO
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 2
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LCRC" begin
    #     announce("4D rules :: 7-point LCRC")
    #     dim = 4
    #     rule = :ns_p7
    #     boundary = :LCRC
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 0
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LCRO" begin
    #     announce("4D rules :: 7-point LCRO")
    #     dim = 4
    #     rule = :ns_p7
    #     boundary = :LCRO
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 1
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LORC" begin
    #     announce("4D rules :: 7-point LORC")
    #     dim = 4
    #     rule = :ns_p7
    #     boundary = :LORC
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 1
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LORO" begin
    #     announce("4D rules :: 7-point LORO")
    #     dim = 4
    #     rule = :ns_p7
    #     boundary = :LORO
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 2
    #     result_string = "4D"
    #     nerr_terms = 3
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f4D, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end
end