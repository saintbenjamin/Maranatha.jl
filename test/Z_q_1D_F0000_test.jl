using .Maranatha.F0000GammaEminus1

@testset "F0000GammaEminus1 1D" begin
    announce("F0000GammaEminus1 1D")

    f(x)  = gtilde_F0000(x; p=3)
    bounds = (0.0, 1.0)

    @testset "Trapezoidal LCRC" begin
        announce("1D rules :: Trapezoidal LCRC")
        dim = 1
        rule = :ns_p2
        boundary = :LCRC
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 40
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "Trapezoidal LCRO" begin
    #     announce("1D rules :: Trapezoidal LCRO")
    #     dim = 1
    #     rule = :ns_p2
    #     boundary = :LCRO
    #     ns = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "Trapezoidal LORC" begin
    #     announce("1D rules :: Trapezoidal LORC")
    #     dim = 1
    #     rule = :ns_p2
    #     boundary = :LORC
    #     ns = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    @testset "Trapezoidal LORO" begin
        announce("1D rules :: Trapezoidal LORO")
        dim = 1
        rule = :ns_p2
        boundary = :LORO
        ns = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        ns .+= 40
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LCRC" begin
        announce("1D rules :: Simpson 1/3 LCRC")
        dim = 1
        rule = :ns_p3
        boundary = :LCRC
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 40
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "Simpson 1/3 LCRO" begin
    #     announce("1D rules :: Simpson 1/3 LCRO")
    #     dim = 1
    #     rule = :ns_p3
    #     boundary = :LCRO
    #     ns = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "Simpson 1/3 LORC" begin
    #     announce("1D rules :: Simpson 1/3 LORC")
    #     dim = 1
    #     rule = :ns_p3
    #     boundary = :LORC
    #     ns = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    @testset "Simpson 1/3 LORO" begin
        announce("1D rules :: Simpson 1/3 LORO")
        dim = 1
        rule = :ns_p3
        boundary = :LORO
        ns = [6, 8, 10, 12, 14, 16, 18, 20, 22]
        ns .+= 40
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LCRC" begin
        announce("1D rules :: Simpson 3/8 LCRC")
        dim = 1
        rule = :ns_p4
        boundary = :LCRC
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 39
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "Simpson 3/8 LCRO" begin
    #     announce("1D rules :: Simpson 3/8 LCRO")
    #     dim = 1
    #     rule = :ns_p4
    #     boundary = :LCRO
    #     ns = [7, 10, 13, 16, 19, 22, 25, 28, 31]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "Simpson 3/8 LORC" begin
    #     announce("1D rules :: Simpson 3/8 LORC")
    #     dim = 1
    #     rule = :ns_p4
    #     boundary = :LORC
    #     ns = [7, 10, 13, 16, 19, 22, 25, 28, 31]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    @testset "Simpson 3/8 LORO" begin
        announce("1D rules :: Simpson 3/8 LORO")
        dim = 1
        rule = :ns_p4
        boundary = :LORO
        ns = [8, 11, 14, 17, 20, 23, 26, 29, 32]
        ns .+= 39
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LCRC" begin
        announce("1D rules :: Bode LCRC")
        dim = 1
        rule = :ns_p5
        boundary = :LCRC
        ns = [8, 12, 16, 20, 24, 28, 32]
        ns .+= 56
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "Bode LCRO" begin
    #     announce("1D rules :: Bode LCRO")
    #     dim = 1
    #     rule = :ns_p5
    #     boundary = :LCRO
    #     ns = [9, 13, 17, 21, 25, 29, 33, 37, 41]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "Bode LORC" begin
    #     announce("1D rules :: Bode LORC")
    #     dim = 1
    #     rule = :ns_p5
    #     boundary = :LORC
    #     ns = [9, 13, 17, 21, 25, 29, 33, 37, 41]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    @testset "Bode LORO" begin
        announce("1D rules :: Bode LORO")
        dim = 1
        rule = :ns_p5
        boundary = :LORO
        ns = [10, 14, 18, 22, 26, 30, 34]
        ns .+= 28
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    # @testset "6-point LCRC" begin
    #     announce("1D rules :: 6-point LCRC")
    #     dim = 1
    #     rule = :ns_p6
    #     boundary = :LCRC
    #     ns = [10, 15, 20, 25, 30, 35, 40]
    #     ns .+= 70
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LORO" begin
    #     announce("1D rules :: 6-point LORO")
    #     dim = 1
    #     rule = :ns_p6
    #     boundary = :LORO
    #     ns = [12, 17, 22, 27, 32, 37, 42]
    #     ns .+= 75
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LCRC" begin
    #     announce("1D rules :: 7-point LCRC")
    #     dim = 1
    #     rule = :ns_p7
    #     boundary = :LCRC
    #     # ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns = [120, 126, 132, 138, 144, 150, 156, 162]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LORO" begin
    #     announce("1D rules :: 7-point LORO")
    #     dim = 1
    #     rule = :ns_p7
    #     boundary = :LORO
    #     # ns = [14, 20, 26, 32, 38, 44, 50, 56, 62]
    #     ns = [122, 128, 134, 140, 146, 152, 158, 164]
    #     result_string = "F0000"
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         f, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end
end

@testset "Integrand preset API (F0000)" begin
    announce("Integrand preset API (F0000)")

    # Registry sanity
    @test :F0000 in Maranatha.Integrands.available_integrands()

    # Construct preset integrand via registry
    f = Maranatha.Integrands.integrand(:F0000; p=3, eps=1e-15)
    bounds = (0.0, 1.0)

    @testset "Simpson 1/3 LCRC (preset)" begin
        announce("1D rules :: Simpson 1/3 LCRC (preset)")
        dim = 1
        rule = :ns_p3
        boundary = :LCRC
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 40
        result_string = "F0000_preset"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end

@testset "Preset vs raw integrand consistency (spot-check)" begin
    announce("Preset vs raw integrand consistency (spot-check)")

    t = 0.37
    f_raw = t -> gtilde_F0000(t; p=3, eps=1e-15)
    f_pre = Maranatha.Integrands.integrand(:F0000; p=3, eps=1e-15)

    @test isfinite(f_raw(t))
    @test isfinite(f_pre(t))

    # Same underlying formula expected (exact match in current design)
    @test f_raw(t) == f_pre(t)
end