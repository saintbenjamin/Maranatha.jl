const NC  = Maranatha.Quadrature.NewtonCotes
const QD  = Maranatha.Quadrature.QuadratureDispatch
const QN  = Maranatha.Quadrature.QuadratureNodes
const EDD = Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative

function _manual_tensor_2d(f, a, b, N, rx, ry, bx, by)
    xs, wx = QN.get_quadrature_1d_nodes_weights(a[1], b[1], N, rx, bx; real_type = Float64)
    ys, wy = QN.get_quadrature_1d_nodes_weights(a[2], b[2], N, ry, by; real_type = Float64)

    total = 0.0
    @inbounds for i in eachindex(xs), j in eachindex(ys)
        total += wx[i] * wy[j] * f(xs[i], ys[j])
    end
    return total
end

function _expected_common_newton_fine_N(rule, boundary, dim, Ntarget)
    Ncand = Ntarget

    while true
        updated = false

        for d in 1:dim
            rd = rule[d]
            bd = boundary[d]
            p = NC._parse_newton_p(rd)
            Nd = NC._nearest_valid_Nsub(p, bd, Ncand)

            if Nd > Ncand
                Ncand = Nd
                updated = true
            end
        end

        updated || return Ncand
    end
end

@testset "Axis-wise rule regression" begin
    @testset "Quadrature uses per-axis rules" begin
        f(x, y) = exp(x) * cos(y) + x * y^2
        a = (0.0, -0.5)
        b = (1.0,  0.75)
        rule = (:gauss_p2, :gauss_p5)
        boundary = (:LU_INEX, :LU_EXEX)
        N = 3

        got = QD.quadrature(f, a, b, N, 2, rule, boundary; real_type = Float64)
        ref = _manual_tensor_2d(f, a, b, N, rule[1], rule[2], boundary[1], boundary[2])

        @test isapprox(got, ref; atol = 1e-12, rtol = 1e-12)
    end

    @testset "Least-chi-square fit uses all rule axes" begin
        f(x, y) = exp(x) * cos(y) + x * y^2
        rule = (:gauss_p2, :gauss_p4)
        boundary = (:LU_INEX, :LU_EXEX)

        res = run_Maranatha(
            f,
            0.0,
            1.0;
            dim = 2,
            nsamples = [2, 3, 4, 5],
            rule = rule,
            boundary = boundary,
            err_method = :forwarddiff,
            nerr_terms = 2,
            fit_terms = 3,
            ff_shift = 0,
            real_type = Float64,
        )

        fit = least_chi_square_fit(res)
        print_fit_result(fit)

        Nref = minimum(res.nsamples)
        need = res.fit_terms + res.ff_shift - 1

        ks_union = unique(sort(vcat(
            EDD._leading_residual_ks_with_center_any(rule[1], boundary[1], Nref; fit_func_terms = need, kmax = 256)[1],
            EDD._leading_residual_ks_with_center_any(rule[2], boundary[2], Nref; fit_func_terms = need, kmax = 256)[1],
        )))

        @test collect(Int.(fit.powers[2:end])) == ks_union[1:(res.fit_terms - 1)]
    end

    @testset "Refinement allows same-family Gauss axis-wise rule" begin
        f(x, y) = exp(x) * cos(y) + x * y^2

        res = run_Maranatha(
            f,
            0.0,
            1.0;
            dim = 2,
            nsamples = [2, 3, 4, 5],
            rule = (:gauss_p3, :gauss_p4),
            boundary = (:LU_INEX, :LU_EXEX),
            err_method = :refinement,
            fit_terms = 3,
            real_type = Float64,
        )

        assert_result_sane(res)
        @test all(e -> e.method == :gauss_refinement_difference, res.err)

        fit = least_chi_square_fit(res)
        @test isfinite(fit.estimate)
        @test isfinite(fit.error_estimate)
    end

    @testset "Refinement Newton-Cotes uses common valid N_fine across axes" begin
        f(x, y) = exp(x) + x * y + y^2

        rule = (:newton_p2, :newton_p3)
        boundary = (:LU_ININ, :LU_EXEX)

        res = run_Maranatha(
            f,
            0.0,
            1.0;
            dim = 2,
            nsamples = [2, 3, 4, 5],
            rule = rule,
            boundary = boundary,
            err_method = :refinement,
            fit_terms = 3,
            real_type = Float64,
        )

        assert_result_sane(res)
        @test all(e -> e.method == :newton_cotes_refinement_difference, res.err)

        for (Ncoarse, e) in zip(res.nsamples, res.err)
            @test e.N_fine == _expected_common_newton_fine_N(rule, boundary, 2, 2Ncoarse)
        end
    end

    @testset "Refinement still blocks mixed-family axis-wise rule" begin
        f(x, y) = exp(x) * cos(y) + x * y^2

        @test_throws ArgumentError run_Maranatha(
            f,
            0.0,
            1.0;
            dim = 2,
            nsamples = [2, 3, 4, 5],
            rule = (:gauss_p3, :newton_p3),
            boundary = :LU_EXEX,
            err_method = :refinement,
            real_type = Float64,
        )
    end
end