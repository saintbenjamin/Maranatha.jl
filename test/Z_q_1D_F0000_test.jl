using .Maranatha.F0000GammaEminus1

ff_tilde(x)  = gtilde_F0000(x; p=4)
ff(x)  = g_F0000_raw(x)

bounds = (0.0, 1.0)
use_threads = false

@testset "F0000GammaEminus1 1D" begin
    announce("F0000GammaEminus1 1D")

    @testset "Gauss LU_EXEX FastDifferentiation.jl" begin
        announce("1D rules :: Gauss LU_EXEX FastDifferentiation.jl")
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
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Gauss LU_EXEX ForwardDiff.jl" begin
        announce("1D rules :: Gauss LU_EXEX ForwardDiff.jl")
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
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff_tilde, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end

    @testset "Gauss LU_EXEX TaylorSeries.jl" begin
        announce("1D rules :: Gauss LU_EXEX TaylorSeries.jl")
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
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
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
        err_method = :fastdifferentiation # :forwarddiff , :taylorseries , :enzyme , :fastdifferentiation
        nerr_terms = 1
        ff_shift = 0
        fit_terms = 3
        result_string = "F0000"
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end

# Registry sanity
@test :F0000 in Maranatha.Integrands.available_integrands()

# Construct preset integrand via registry
ff_here = Maranatha.Integrands.integrand(:F0000; p=4, eps=1e-15)
bounds = (0.0, 1.0)
use_threads = false

@testset "Integrand preset API (F0000)" begin
    announce("Integrand preset API (F0000)")

    @testset "Gauss LU_EXEX (preset)" begin
        announce("1D rules :: Gauss LU_EXEX (preset)")
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
        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff_here, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=err_method, fit_terms=fit_terms, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
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