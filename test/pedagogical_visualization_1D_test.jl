using .Maranatha.F0000GammaEminus1

@testset "Pedagogical visualization of 1D rules" begin
    announce("Pedagogical visualization of 1D rules")

    # ff(x) = exp(-x) * cos(6x)
    # ff(x) = sin(x)
    # bounds = (0.0, π)
    # ff(x) = 4 - x^2
    # bounds = (0.0, 1)

    # dim = 1
    # rule = :newton_p7
    # boundary = :LU_ININ
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

    use_threads = false

    dim = 1
    rule = :newton_p4
    boundary = :LU_EXEX
    # rule = :gauss_p3
    # boundary = :LU_EXEX
    # rule = :bspline_interp_p3
    # boundary = :LU_ININ
    ns = 8
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