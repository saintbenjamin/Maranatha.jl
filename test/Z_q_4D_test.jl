@testset "Z_q 4D" begin
    announce("Z_q 4D")

    @testset "Gauss 2-point LU_EXEX" begin
        announce("4D rules :: Gauss 2-point LU_EXEX")
        dim = 4

        rule = :gauss_p2
        boundary = :LU_EXEX
        ns = [2, 3, 4, 5, 6, 7, 8, 9]
        ns .+= 20
        result_string = "Z_q"
        nerr_terms = 1
        ff_shift = 0

        ff(x1,x2,x3,x4) = Maranatha.Z_q.integrand_Z_q((x1,x2,x3,x4))
        bounds=(0.0,π)

        est, fit, res = Maranatha.Runner.run_Maranatha(
            ff, bounds...; dim=dim, nsamples=ns,
            rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift
        )

        assert_result_sane(res); @test isfinite(est)
        maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
    end
end