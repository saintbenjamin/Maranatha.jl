@testset "Pedagogical visualization of 1D rules" begin
    ff(x) = exp(-x) * cos(6x)
    bounds = (0.0, π)
    ns = 8
    rule = :gauss_p3
    boundary = :LU_EXEX
    result_string = "demo"
    save_file = false

    plot_quadrature_coverage_1d(
        ff, 
        bounds..., 
        ns;
        rule=rule,
        boundary=boundary,
        name=result_string,
        save_file=save_file
    )

end