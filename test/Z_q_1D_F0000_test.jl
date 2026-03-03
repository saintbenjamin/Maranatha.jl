using .Maranatha.F0000GammaEminus1

@testset "F0000GammaEminus1 1D" begin
    announce("F0000GammaEminus1 1D")

    ff(x)  = gtilde_F0000(x; p=3)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Trapezoidal LU_ININ" begin
        announce("1D rules :: Trapezoidal LU_ININ")
        dim = 1
        rule = :newton_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8]
        ns .+= 0
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_INEX" begin
        announce("1D rules :: Trapezoidal LU_INEX")
        dim = 1
        rule = :newton_p2
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8]
        ns .+= 1
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_EXIN" begin
        announce("1D rules :: Trapezoidal LU_EXIN")
        dim = 1
        rule = :newton_p2
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8]
        ns .+= 1
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_EXEX" begin
        announce("1D rules :: Trapezoidal LU_EXEX")
        dim = 1
        rule = :newton_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8]
        ns .+= 2
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_ININ" begin
        announce("1D rules :: Simpson 1/3 LU_ININ")
        dim = 1
        rule = :newton_p3
        boundary = :LU_ININ
        ns = [4, 6, 8, 10, 12, 14, 16]
        ns .+= 0
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_INEX" begin
        announce("1D rules :: Simpson 1/3 LU_INEX")
        dim = 1
        rule = :newton_p3
        boundary = :LU_INEX
        ns = [4, 6, 8, 10, 12, 14, 16]
        ns .+= 1
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_EXIN" begin
        announce("1D rules :: Simpson 1/3 LU_EXIN")
        dim = 1
        rule = :newton_p3
        boundary = :LU_EXIN
        ns = [4, 6, 8, 10, 12, 14, 16]
        ns .+= 1
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_EXEX" begin
        announce("1D rules :: Simpson 1/3 LU_EXEX")
        dim = 1
        rule = :newton_p3
        boundary = :LU_EXEX
        ns = [4, 6, 8, 10, 12, 14, 16]
        ns .+= 2
        ns .+= 40
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_ININ" begin
        announce("1D rules :: Simpson 3/8 LU_ININ")
        dim = 1
        rule = :newton_p4
        boundary = :LU_ININ
        ns = [6, 9, 12, 15, 18, 21, 24]
        ns .+= 0
        ns .+= 39
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_INEX" begin
        announce("1D rules :: Simpson 3/8 LU_INEX")
        dim = 1
        rule = :newton_p4
        boundary = :LU_INEX
        ns = [6, 9, 12, 15, 18, 21, 24]
        ns .+= 1
        ns .+= 39
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_EXIN" begin
        announce("1D rules :: Simpson 3/8 LU_EXIN")
        dim = 1
        rule = :newton_p4
        boundary = :LU_EXIN
        ns = [6, 9, 12, 15, 18, 21, 24]
        ns .+= 1
        ns .+= 39
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_EXEX" begin
        announce("1D rules :: Simpson 3/8 LU_EXEX")
        dim = 1
        rule = :newton_p4
        boundary = :LU_EXEX
        ns = [6, 9, 12, 15, 18, 21, 24]
        ns .+= 2
        ns .+= 39
        result_string = "F0000"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_ININ" begin
        announce("1D rules :: Bode LU_ININ")
        dim = 1
        rule = :newton_p5
        boundary = :LU_ININ
        ns = [8, 12, 16, 20, 24, 28, 32]
        ns .+= 0
        ns .+= 36
        result_string = "F0000"
        nerr_terms = 2
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=2, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

#=     @testset "Bode LU_INEX" begin
        announce("1D rules :: Bode LU_INEX")
        dim = 1
        rule = :newton_p5
        boundary = :LU_INEX
        ns = [8, 12, 16, 20, 24, 28, 32]
        ns .+= 1
        ns .+= 56
        result_string = "F0000"
        nerr_terms = 2
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_EXIN" begin
        announce("1D rules :: Bode LU_EXIN")
        dim = 1
        rule = :newton_p5
        boundary = :LU_EXIN
        ns = [8, 12, 16, 20, 24, 28, 32]
        ns .+= 1
        ns .+= 56
        result_string = "F0000"
        nerr_terms = 2
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_EXEX" begin
        announce("1D rules :: Bode LU_EXEX")
        dim = 1
        rule = :newton_p5
        boundary = :LU_EXEX
        ns = [8, 12, 16, 20, 24, 28, 32]
        ns .+= 2
        ns .+= 56
        result_string = "F0000"
        nerr_terms = 2
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end =#

    # @testset "6-point LU_ININ" begin
    #     announce("1D rules :: 6-point LU_ININ")
    #     dim = 1
    #     rule = :newton_p6
    #     boundary = :LU_ININ
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 0
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LU_INEX" begin
    #     announce("1D rules :: 6-point LU_INEX")
    #     dim = 1
    #     rule = :newton_p6
    #     boundary = :LU_INEX
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 1
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LU_EXIN" begin
    #     announce("1D rules :: 6-point LU_EXIN")
    #     dim = 1
    #     rule = :newton_p6
    #     boundary = :LU_EXIN
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 1
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "6-point LU_EXEX" begin
    #     announce("1D rules :: 6-point LU_EXEX")
    #     dim = 1
    #     rule = :newton_p6
    #     boundary = :LU_EXEX
    #     ns = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     ns .+= 2
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LU_ININ" begin
    #     announce("1D rules :: 7-point LU_ININ")
    #     dim = 1
    #     rule = :newton_p7
    #     boundary = :LU_ININ
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 0
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LU_INEX" begin
    #     announce("1D rules :: 7-point LU_INEX")
    #     dim = 1
    #     rule = :newton_p7
    #     boundary = :LU_INEX
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 1
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LU_EXIN" begin
    #     announce("1D rules :: 7-point LU_EXIN")
    #     dim = 1
    #     rule = :newton_p7
    #     boundary = :LU_EXIN
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 1
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end

    # @testset "7-point LU_EXEX" begin
    #     announce("1D rules :: 7-point LU_EXEX")
    #     dim = 1
    #     rule = :newton_p7
    #     boundary = :LU_EXEX
    #     ns = [12, 18, 24, 30, 36, 42, 48, 54, 60]
    #     ns .+= 2
    #     result_string = "F0000"
    #     nerr_terms = 1
    #     ff_shift = 0
    #     est, fit, res = Maranatha.Runner.run_Maranatha(
    #         ff, bounds...; dim=dim, nsamples=ns,
    #         rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
    #     )
    #     assert_result_sane(res); @test isfinite(est)
    #     maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    # end
end

#= @testset "Integrand preset API (F0000)" begin
    announce("Integrand preset API (F0000)")

    # Registry sanity
    @test :F0000 in Maranatha.Integrands.available_integrands()

    # Construct preset integrand via registry
    ff = Maranatha.Integrands.integrand(:F0000; p=3, eps=1e-15)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Simpson 1/3 LU_ININ (preset)" begin
        announce("1D rules :: Simpson 1/3 LU_ININ (preset)")
        dim = 1
        rule = :newton_p3
        boundary = :LU_ININ
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 0
        ns .+= 40
        result_string = "F0000_preset"
        nerr_terms = 2
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end =#

#= @testset "Preset vs raw integrand consistency (spot-check)" begin
    announce("Preset vs raw integrand consistency (spot-check)")

    t = 0.37
    f_raw = t -> gtilde_F0000(t; p=3, eps=1e-15)
    f_pre = Maranatha.Integrands.integrand(:F0000; p=3, eps=1e-15)

    @test isfinite(f_raw(t))
    @test isfinite(f_pre(t))

    # Same underlying formula expected (exact match in current design)
    @test f_raw(t) == f_pre(t)
end =#