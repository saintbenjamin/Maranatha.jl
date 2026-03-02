using .Maranatha.F0000GammaEminus1

@testset "1D rules" begin
    announce("1D rules")

    f1D(x) = sin(x)
    bounds = (0.0, π)
    # f2D(x, y) = exp(-x^2 - y^2)
    # bounds = (0.0, 1.0)
    # ff(x)  = gtilde_F0000(x; p=3)
    # bounds = (0.0, 1.0)
    use_threads = true

    @testset "1D Gauss LORO" begin
        announce("1D rules :: Gauss LORO")
        dim = 1
        rule = :gauss_p2
        boundary = :LORO
        ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        ns .+= 0
        result_string = "Gauss_1D"
        nerr_terms = 3
        ff_shift = 0
        est, fit, res = Maranatha.Runner.run_Maranatha(
            f1D, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift, use_threads=use_threads
        )
        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end