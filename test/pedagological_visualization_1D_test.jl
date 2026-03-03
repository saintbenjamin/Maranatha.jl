using .Maranatha.F0000GammaEminus1

@testset "Pedagological visualization of 1D rules" begin
    announce("Pedagological visualization of 1D rules")

    # f(x) = exp(-x) * cos(6x)
    # # f(x) = sin(x)
    # bounds = (0.0, π)
    # # f(x) = x
    # # bounds = (0.0, 10)

    # dim = 1
    # rule = :ns_p7
    # boundary = :LCRC
    # ns = 24
    # result_string = "PV_1D"
    # nerr_terms = 3
    # ff_shift = 0
    # plot_quadrature_coverage_1d(
    #     f, bounds..., ns;
    #     rule=rule,
    #     boundary=boundary,
    #     name="demo1"
    # )

    ff(x)  = gtilde_F0000(x; p=3)
    bounds = (0.0, 1.0)
    use_threads = true

    dim = 1
    rule = :bsplI_p3
    boundary = :LCRC
    ns = 48
    result_string = "PV_1D"
    nerr_terms = 3
    ff_shift = 0
    plot_quadrature_coverage_1d(
        ff, bounds..., ns;
        rule=rule,
        boundary=boundary,
        name="demo1"
    )

end