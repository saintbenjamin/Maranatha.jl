# ============================================================================
# test/runtests.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

using Test

# Project-local includes (same as your entry script)
include(joinpath(@__DIR__, "..", "src", "Maranatha.jl"))
using .Maranatha

include(joinpath(@__DIR__, "..", "src", "figs", "PlotTools.jl"))
using .PlotTools

include(joinpath(@__DIR__, "..", "src", "functions", "F0000GammaEminus1.jl"))
using .F0000GammaEminus1

# ----------------------------------------------------------------------------
# Optional plotting switch:
#   MARANATHA_PLOT=1  -> enable plots
#   (default)         -> compute everything but do not call plot_convergence_result
# ----------------------------------------------------------------------------
const DO_PLOT = get(ENV, "MARANATHA_PLOT", "0") == "0"

function maybe_plot(tag::AbstractString, h, avg, err, fit; rule::Symbol)
    if DO_PLOT
        plot_convergence_result(tag, h, avg, err, fit; rule=rule)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Smoke checks: basic sanity without assuming exact numbers
# ----------------------------------------------------------------------------
function assert_result_sane(res)
    @test length(res.h) == length(res.avg) == length(res.err)
    @test all(isfinite, res.h)
    @test all(isfinite, res.avg)
    @test all(isfinite, res.err)
    @test all(>(0.0), res.h)  # step sizes should be positive

    # NOTE:
    # res.err can be signed depending on the error estimator / fitting pipeline.
    # We only require it to be finite here.
    return nothing
end

function announce(title::AbstractString)
    println("▶ ", title)
    println()
    return nothing
end

@testset "Maranatha regression suite" begin
    announce("Maranatha regression suite")

    # ==============================================================
    # use one fit function term in fit_convergence()
    # ==============================================================

    @testset "F0000GammaEminus1 1D (single-term fit)" begin
        announce("F0000GammaEminus1 1D (single-term fit)")

        # Your "f1d" integrand for Maranatha: y ↦ g(y)
        f1d(x)  = gtilde_F0000(x; p=3)
        f1d4(x) = gtilde_F0000(x; p=4)

        ns_3 = [30, 33, 36, 39, 42, 45, 48]
        ns_4 = [40, 44, 48, 52, 56, 60, 64]

        bounds = (0.0, 1.0)

        @testset "1D Simpson 1/3 Close" begin
            announce("1D Simpson 1/3 Close")
            est1, fit1, res1 = Maranatha.Runner.run_Maranatha(
                f1d, bounds...; dim=1, nsamples=ns_4,
                rule=:simpson13_close, err_method=:derivative, fit_terms=2
            )
            assert_result_sane(res1)
            @test isfinite(est1)
            maybe_plot("1D", res1.h, res1.avg, res1.err, fit1; rule=:simpson13_close)
        end

        @testset "1D Simpson 3/8 Close" begin
            announce("1D Simpson 3/8 Close")
            est2, fit2, res2 = Maranatha.Runner.run_Maranatha(
                f1d, bounds...; dim=1, nsamples=ns_3,
                rule=:simpson38_close, err_method=:derivative, fit_terms=2
            )
            assert_result_sane(res2)
            @test isfinite(est2)
            maybe_plot("1D", res2.h, res2.avg, res2.err, fit2; rule=:simpson38_close)
        end

        @testset "1D Bode Close" begin
            announce("1D Bode Close")
            est3, fit3, res3 = Maranatha.Runner.run_Maranatha(
                f1d4, bounds...; dim=1, nsamples=ns_4,
                rule=:bode_close, err_method=:derivative, fit_terms=2
            )
            assert_result_sane(res3)
            @test isfinite(est3)
            maybe_plot("1D", res3.h, res3.avg, res3.err, fit3; rule=:bode_close)
        end
    end

    # ==============================================================
    # use three fit function terms in fit_convergence()
    # ==============================================================

    @testset "Canonical integrands (multi-dim, multi-rule)" begin
        announce("Canonical integrands (multi-dim, multi-rule)")

        f1d(x) = sin(x)
        f2d(x, y) = exp(-x^2 - y^2)
        f3d(x, y, z) = exp(-x^2 - y^2 - z^2)
        f4d(x, y, z, t) = x * y * z * t

        bounds = (0.0, 1.0)
        # bounds = (0.0, π)

        ns   = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns_3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

        ns_13_open   = [8, 12, 16, 20, 24, 28, 32, 36, 40]
        ns_bode_open = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]

        # ----------------------------
        # 1D
        # ----------------------------
        @testset "1D rules" begin
            announce("1D rules")

            @testset "Simpson 1/3 Close" begin
                announce("1D rules :: Simpson 1/3 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns,
                    rule=:simpson13_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:simpson13_close)
            end

            @testset "Simpson 1/3 Open" begin
                announce("1D rules :: Simpson 1/3 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns_13_open,
                    rule=:simpson13_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:simpson13_open)
            end

            @testset "Simpson 3/8 Close" begin
                announce("1D rules :: Simpson 3/8 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns_3,
                    rule=:simpson38_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:simpson38_close)
            end

            @testset "Simpson 3/8 Open" begin
                announce("1D rules :: Simpson 3/8 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns,
                    rule=:simpson38_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:simpson38_open)
            end

            @testset "Bode Close" begin
                announce("1D rules :: Bode Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns,
                    rule=:bode_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:bode_close)
            end

            @testset "Bode Open" begin
                announce("1D rules :: Bode Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f1d, bounds...; dim=1, nsamples=ns_bode_open,
                    rule=:bode_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("1D", res.h, res.avg, res.err, fit; rule=:bode_open)
            end
        end

        # ----------------------------
        # 2D
        # ----------------------------
        @testset "2D rules" begin
            announce("2D rules")

            @testset "Simpson 1/3 Close" begin
                announce("2D rules :: Simpson 1/3 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns,
                    rule=:simpson13_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:simpson13_close)
            end

            @testset "Simpson 1/3 Open (richardson)" begin
                announce("2D rules :: Simpson 1/3 Open (richardson)")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns_13_open,
                    rule=:simpson13_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:simpson13_open)
            end

            @testset "Simpson 3/8 Close" begin
                announce("2D rules :: Simpson 3/8 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns_3,
                    rule=:simpson38_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:simpson38_close)
            end

            @testset "Simpson 3/8 Open" begin
                announce("2D rules :: Simpson 3/8 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns,
                    rule=:simpson38_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:simpson38_open)
            end

            @testset "Bode Close" begin
                announce("2D rules :: Bode Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns,
                    rule=:bode_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:bode_close)
            end

            @testset "Bode Open" begin
                announce("2D rules :: Bode Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f2d, bounds...; dim=2, nsamples=ns_bode_open,
                    rule=:bode_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("2D", res.h, res.avg, res.err, fit; rule=:bode_open)
            end
        end

        # ----------------------------
        # 3D
        # ----------------------------
        @testset "3D rules" begin
            announce("3D rules")

            @testset "Simpson 1/3 Close" begin
                announce("3D rules :: Simpson 1/3 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns,
                    rule=:simpson13_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:simpson13_close)
            end

            @testset "Simpson 1/3 Open" begin
                announce("3D rules :: Simpson 1/3 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns_13_open,
                    rule=:simpson13_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:simpson13_open)
            end

            @testset "Simpson 3/8 Close" begin
                announce("3D rules :: Simpson 3/8 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns_3,
                    rule=:simpson38_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:simpson38_close)
            end

            @testset "Simpson 3/8 Open" begin
                announce("3D rules :: Simpson 3/8 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns,
                    rule=:simpson38_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:simpson38_open)
            end

            @testset "Bode Close" begin
                announce("3D rules :: Bode Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns,
                    rule=:bode_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:bode_close)
            end

            @testset "Bode Open" begin
                announce("3D rules :: Bode Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f3d, bounds...; dim=3, nsamples=ns_bode_open,
                    rule=:bode_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("3D", res.h, res.avg, res.err, fit; rule=:bode_open)
            end
        end

        # ----------------------------
        # 4D
        # ----------------------------
        @testset "4D rules" begin
            announce("4D rules")

            @testset "Simpson 1/3 Close" begin
                announce("4D rules :: Simpson 1/3 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns,
                    rule=:simpson13_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:simpson13_close)
            end

            @testset "Simpson 1/3 Open" begin
                announce("4D rules :: Simpson 1/3 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns_13_open,
                    rule=:simpson13_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:simpson13_open)
            end

            @testset "Simpson 3/8 Close" begin
                announce("4D rules :: Simpson 3/8 Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns_3,
                    rule=:simpson38_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:simpson38_close)
            end

            @testset "Simpson 3/8 Open" begin
                announce("4D rules :: Simpson 3/8 Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns,
                    rule=:simpson38_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:simpson38_open)
            end

            @testset "Bode Close" begin
                announce("4D rules :: Bode Close")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns,
                    rule=:bode_close, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:bode_close)
            end

            @testset "Bode Open" begin
                announce("4D rules :: Bode Open")
                est, fit, res = Maranatha.Runner.run_Maranatha(
                    f4d, bounds...; dim=4, nsamples=ns_bode_open,
                    rule=:bode_open, err_method=:derivative, fit_terms=4
                )
                assert_result_sane(res); @test isfinite(est)
                maybe_plot("4D", res.h, res.avg, res.err, fit; rule=:bode_open)
            end
        end
    end
end