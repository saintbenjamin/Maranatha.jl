@testset "4D rules" begin
    announce("4D rules")

    ff(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Trapezoidal LU_ININ" begin
        announce("4D rules :: Trapezoidal LU_ININ")
        dim = 4
        rule = :newton_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_INEX" begin
        announce("4D rules :: Trapezoidal LU_INEX")
        dim = 4
        rule = :newton_p2
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_EXIN" begin
        announce("4D rules :: Trapezoidal LU_EXIN")
        dim = 4
        rule = :newton_p2
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Trapezoidal LU_EXEX" begin
        announce("4D rules :: Trapezoidal LU_EXEX")
        dim = 4
        rule = :newton_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 2
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_ININ" begin
        announce("4D rules :: Simpson 1/3 LU_ININ")
        dim = 4
        rule = :newton_p3
        boundary = :LU_ININ
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_INEX" begin
        announce("4D rules :: Simpson 1/3 LU_INEX")
        dim = 4
        rule = :newton_p3
        boundary = :LU_INEX
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_EXIN" begin
        announce("4D rules :: Simpson 1/3 LU_EXIN")
        dim = 4
        rule = :newton_p3
        boundary = :LU_EXIN
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 1/3 LU_EXEX" begin
        announce("4D rules :: Simpson 1/3 LU_EXEX")
        dim = 4
        rule = :newton_p3
        boundary = :LU_EXEX
        ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        ns .+= 2
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_ININ" begin
        announce("4D rules :: Simpson 3/8 LU_ININ")
        dim = 4
        rule = :newton_p4
        boundary = :LU_ININ
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_INEX" begin
        announce("4D rules :: Simpson 3/8 LU_INEX")
        dim = 4
        rule = :newton_p4
        boundary = :LU_INEX
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_EXIN" begin
        announce("4D rules :: Simpson 3/8 LU_EXIN")
        dim = 4
        rule = :newton_p4
        boundary = :LU_EXIN
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Simpson 3/8 LU_EXEX" begin
        announce("4D rules :: Simpson 3/8 LU_EXEX")
        dim = 4
        rule = :newton_p4
        boundary = :LU_EXEX
        ns = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        ns .+= 2
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_ININ" begin
        announce("4D rules :: Bode LU_ININ")
        dim = 4
        rule = :newton_p5
        boundary = :LU_ININ
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 0
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_INEX" begin
        announce("4D rules :: Bode LU_INEX")
        dim = 4
        rule = :newton_p5
        boundary = :LU_INEX
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_EXIN" begin
        announce("4D rules :: Bode LU_EXIN")
        dim = 4
        rule = :newton_p5
        boundary = :LU_EXIN
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 1
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Bode LU_EXEX" begin
        announce("4D rules :: Bode LU_EXEX")
        dim = 4
        rule = :newton_p5
        boundary = :LU_EXEX
        ns = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns .+= 2
        err_method = :forwarddiff # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 3
        ff_shift = 0
        fit_terms = 4
        result_string = "4D"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end