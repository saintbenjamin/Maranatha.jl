@testset "4D rules" begin
    announce("4D rules")

    f4D(x, y, z, t) = sin(x * y^3 * z * t) * exp(x^2)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Gauss 2-point LU_ININ" begin
        announce("4D rules :: Gauss 2-point LU_ININ")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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


    @testset "Gauss 2-point LU_INEX" begin
        announce("4D rules :: Gauss 2-point LU_INEX")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 2-point LU_EXIN" begin
        announce("4D rules :: Gauss 2-point LU_EXIN")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 2-point LU_EXEX" begin
        announce("4D rules :: Gauss 2-point LU_EXEX")
        dim = 4
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 3-point LU_ININ" begin
        announce("4D rules :: Gauss 3-point LU_ININ")
        dim = 4
        rule = :gauss_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 3-point LU_INEX" begin
        announce("4D rules :: Gauss 3-point LU_INEX")
        dim = 4
        rule = :gauss_p3
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 3-point LU_EXIN" begin
        announce("4D rules :: Gauss 3-point LU_EXIN")
        dim = 4
        rule = :gauss_p3
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 3-point LU_EXEX" begin
        announce("4D rules :: Gauss 3-point LU_EXEX")
        dim = 4
        rule = :gauss_p3
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 4-point LU_ININ" begin
        announce("4D rules :: Gauss 4-point LU_ININ")
        dim = 4
        rule = :gauss_p4
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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


    @testset "Gauss 4-point LU_INEX" begin
        announce("4D rules :: Gauss 4-point LU_INEX")
        dim = 4
        rule = :gauss_p4
        boundary = :LU_INEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 4-point LU_EXIN" begin
        announce("4D rules :: Gauss 4-point LU_EXIN")
        dim = 4
        rule = :gauss_p4
        boundary = :LU_EXIN
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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

    @testset "Gauss 4-point LU_EXEX" begin
        announce("4D rules :: Gauss 4-point LU_EXEX")
        dim = 4
        rule = :gauss_p4
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
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
end