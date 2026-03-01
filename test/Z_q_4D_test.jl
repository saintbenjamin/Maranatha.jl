@testset "Z_q 4D" begin
    announce("Z_q 4D")
    dim = 4

    # rule = :ns_p2
    # boundary = :LORO
    # ns = [2, 3, 4, 5, 6, 7, 8]
    # ns .+= 2
    # nerr_terms = 2
    # ff_shift = 0

    rule = :ns_p3
    boundary = :LORO
    ns = [4, 6, 8, 10, 12, 14, 16]
    ns .+= 2
    # ns .+= 10
    nerr_terms = 2
    ff_shift = 0

    # rule = :ns_p4
    # boundary = :LORO
    # ns = [6, 9, 12, 15, 18, 21, 24]
    # ns .+= 2
    # # ns .+= 18
    # nerr_terms = 2
    # ff_shift = 0

    # rule = :ns_p5
    # boundary = :LORO
    # ns = [8, 12, 16, 20, 24, 28, 32]
    # ns .+= 2
    # # ns .+= 8
    # nerr_terms = 2
    # ff_shift = 0

    ff(x1,x2,x3,x4) = Maranatha.Z_q.integrand_Z_q((x1,x2,x3,x4))
    # display(f(π/2,π/2,π/2,π/2))
    # bounds=(-π,π)
    bounds=(0.0,π)
    result_string = "Z_q"

    est, fit, res = Maranatha.Runner.run_Maranatha(
        ff, bounds...; dim=dim, nsamples=ns,
        rule=rule, boundary=boundary, err_method=:derivative, fit_terms=4, nerr_terms=nerr_terms, ff_shift=ff_shift
    )

    assert_result_sane(res); @test isfinite(est)
    maybe_plot(bounds..., result_string, res.h, res.avg, res.err, fit; rule=rule, boundary=boundary)
end