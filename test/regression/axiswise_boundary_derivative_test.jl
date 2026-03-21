using Test
using Maranatha

const EDD = Maranatha.ErrorEstimate.ErrorDispatch.ErrorDispatchDerivative

function _merged_terms_from_nd(err_nd)
    ks_all = Int[]
    for axis_res in err_nd.per_axis
        append!(ks_all, axis_res.ks)
    end
    sort!(unique!(ks_all))

    T = eltype(err_nd.per_axis[1].terms)
    terms = zeros(T, length(ks_all))

    for axis_res in err_nd.per_axis
        for j in eachindex(axis_res.ks)
            k = axis_res.ks[j]
            i = findfirst(==(k), ks_all)
            i === nothing && continue
            terms[i] += axis_res.terms[j]
        end
    end

    return ks_all, terms
end

function _assert_wrapper_matches_nd(flat_res, nd_res; atol = 1e-11, rtol = 1e-11)
    @test hasproperty(flat_res, :per_axis)
    @test length(flat_res.per_axis) == length(nd_res.per_axis)

    for axis in eachindex(nd_res.per_axis)
        got = flat_res.per_axis[axis]
        ref = nd_res.per_axis[axis]

        @test got.ks == ref.ks
        @test isapprox(collect(Float64.(got.coeffs)), collect(Float64.(ref.coeffs)); atol = atol, rtol = rtol)
        @test isapprox(collect(Float64.(got.derivatives)), collect(Float64.(ref.derivatives)); atol = atol, rtol = rtol)
        @test isapprox(collect(Float64.(got.terms)), collect(Float64.(ref.terms)); atol = atol, rtol = rtol)
        @test isapprox(float(got.total), float(ref.total); atol = atol, rtol = rtol)
    end

    ks_expected, terms_expected = _merged_terms_from_nd(nd_res)

    @test flat_res.ks == ks_expected
    @test isapprox(collect(Float64.(flat_res.terms)), collect(Float64.(terms_expected)); atol = atol, rtol = rtol)
    @test isapprox(float(flat_res.total), float(nd_res.total); atol = atol, rtol = rtol)
    @test isapprox(collect(Float64.(flat_res.center)), collect(Float64.(nd_res.center)); atol = atol, rtol = rtol)
    @test isapprox(collect(Float64.(flat_res.h)), collect(Float64.(nd_res.h)); atol = atol, rtol = rtol)
end

@testset "Axis-wise boundary derivative regression" begin
    @testset "Direct wrappers agree with generic nd" begin
        f2(x, y) = exp(x) * cos(y) + x * y^2
        b2 = (:LU_INEX, :LU_EXEX)

        got2 = EDD.error_estimate_derivative_direct_2d(
            f2, 0.0, 1.0, 3, :gauss_p3, b2;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref2 = EDD.error_estimate_derivative_direct_nd(
            f2, 0.0, 1.0, 3, :gauss_p3, b2;
            dim = 2,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got2, ref2)

        f3(x, y, z) = exp(x) * cos(y) + z^2 * x - sin(x * z)
        b3 = (:LU_INEX, :LU_EXEX, :LU_EXIN)

        got3 = EDD.error_estimate_derivative_direct_3d(
            f3, 0.0, 1.0, 3, :gauss_p3, b3;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref3 = EDD.error_estimate_derivative_direct_nd(
            f3, 0.0, 1.0, 3, :gauss_p3, b3;
            dim = 3,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got3, ref3)

        f4(x, y, z, t) = exp(x) + x * y - z * t + sin(y + z)
        b4 = (:LU_INEX, :LU_EXEX, :LU_EXIN, :LU_INEX)

        got4 = EDD.error_estimate_derivative_direct_4d(
            f4, 0.0, 1.0, 2, :gauss_p3, b4;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref4 = EDD.error_estimate_derivative_direct_nd(
            f4, 0.0, 1.0, 2, :gauss_p3, b4;
            dim = 4,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got4, ref4)
    end

    @testset "Jet wrappers agree with generic nd" begin
        f2(x, y) = exp(x) * cos(y) + x * y^2
        b2 = (:LU_INEX, :LU_EXEX)

        got2 = EDD.error_estimate_derivative_jet_2d(
            f2, 0.0, 1.0, 3, :gauss_p3, b2;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref2 = EDD.error_estimate_derivative_jet_nd(
            f2, 0.0, 1.0, 3, :gauss_p3, b2;
            dim = 2,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got2, ref2)

        f3(x, y, z) = exp(x) * cos(y) + z^2 * x - sin(x * z)
        b3 = (:LU_INEX, :LU_EXEX, :LU_EXIN)

        got3 = EDD.error_estimate_derivative_jet_3d(
            f3, 0.0, 1.0, 3, :gauss_p3, b3;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref3 = EDD.error_estimate_derivative_jet_nd(
            f3, 0.0, 1.0, 3, :gauss_p3, b3;
            dim = 3,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got3, ref3)

        f4(x, y, z, t) = exp(x) + x * y - z * t + sin(y + z)
        b4 = (:LU_INEX, :LU_EXEX, :LU_EXIN, :LU_INEX)

        got4 = EDD.error_estimate_derivative_jet_4d(
            f4, 0.0, 1.0, 2, :gauss_p3, b4;
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        ref4 = EDD.error_estimate_derivative_jet_nd(
            f4, 0.0, 1.0, 2, :gauss_p3, b4;
            dim = 4,
            err_method = :forwarddiff,
            nerr_terms = 2,
            kmax = 32,
            real_type = Float64,
        )
        _assert_wrapper_matches_nd(got4, ref4)
    end

    @testset "Least-chi-square fit uses all boundary axes" begin
        f(x, y) = exp(x) * cos(y) + x * y^2
        boundary = (:LU_INEX, :LU_EXEX)

        res = run_Maranatha(
            f,
            0.0,
            1.0;
            dim = 2,
            nsamples = [2, 3, 4, 5],
            rule = :gauss_p3,
            boundary = boundary,
            err_method = :forwarddiff,
            nerr_terms = 2,
            fit_terms = 3,
            ff_shift = 0,
            real_type = Float64,
        )

        assert_result_sane(res)

        fit = least_chi_square_fit(res)

        Nref = minimum(res.nsamples)
        need = res.fit_terms + res.ff_shift - 1

        ks_union = unique(sort(vcat(
            EDD._leading_residual_ks_with_center_any(
                res.rule, boundary[1], Nref;
                fit_func_terms = need,
                kmax = 256,
            )[1],
            EDD._leading_residual_ks_with_center_any(
                res.rule, boundary[2], Nref;
                fit_func_terms = need,
                kmax = 256,
            )[1],
        )))

        @test collect(Int.(fit.powers[2:end])) == ks_union[1:(res.fit_terms - 1)]
    end
end