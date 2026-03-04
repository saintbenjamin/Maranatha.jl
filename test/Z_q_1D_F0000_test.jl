using .Maranatha.F0000GammaEminus1

ff(x)  = gtilde_F0000(x; p=4)
bounds = (0.0, 1.0)
use_threads = false

@testset "F0000GammaEminus1 1D" begin
    announce("F0000GammaEminus1 1D")

    @testset "Gauss LU_EXEX" begin
        announce("1D rules :: Gauss LU_EXEX")
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        result_string = "F0000"
        nerr_terms = 1
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "B-spline 3-point LU_ININ" begin
        announce("1D rules :: B-spline 3-point LU_ININ")
        dim = 1
        rule = :bspline_interp_p3
        boundary = :LU_ININ
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 35
        result_string = "F0000"
        nerr_terms = 1
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end

@testset "Integrand preset API (F0000)" begin
    announce("Integrand preset API (F0000)")

    # Registry sanity
    @test :F0000 in Maranatha.Integrands.available_integrands()

    # Construct preset integrand via registry
    ff = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)
    bounds = (0.0, 1.0)
    use_threads = false

    @testset "Gauss LU_EXEX (preset)" begin
        announce("1D rules :: Gauss LU_EXEX (preset)")
        dim = 1
        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        result_string = "F0000_preset"
        nerr_terms = 1
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=3, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end

@testset "Preset vs raw integrand consistency (spot-check)" begin
    announce("Preset vs raw integrand consistency (spot-check)")

    t = 0.37
    f_raw = t -> gtilde_F0000(t; p=4, eps=1e-15)
    f_pre = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)

    @test isfinite(f_raw(t))
    @test isfinite(f_pre(t))

    # Same underlying formula expected (exact match in current design)
    @test f_raw(t) == f_pre(t)
end