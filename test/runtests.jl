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

# include(joinpath(@__DIR__, "..", "src", "figs", "PlotTools.jl"))
# using .PlotTools

using .Maranatha.PlotTools

# ----------------------------------------------------------------------------
# Optional plotting switch:
#   MARANATHA_PLOT=1  -> enable plots
#   (default)         -> compute everything but do not call plot_convergence_result
# ----------------------------------------------------------------------------
const DO_PLOT = get(ENV, "MARANATHA_PLOT", "0") == "0"

function maybe_plot(a, b, tag::AbstractString, h, avg, err, fit; rule::Symbol, boundary::Symbol)
    if DO_PLOT
        plot_convergence_result(a, b, tag, h, avg, err, fit; rule=rule, boundary=boundary)
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
    println()
    println("▶ ", title)
    println()
    return nothing
end

@testset "Maranatha regression suite" begin
    announce("Maranatha regression suite")

    include("Z_q_1D_F0000_test.jl")

    @testset "Canonical integrands (multi-dim, multi-rule)" begin
        announce("Canonical integrands (multi-dim, multi-rule)")
        include("canonical_1D_test.jl")
        include("canonical_2D_test.jl")
        include("canonical_3D_test.jl")
        include("canonical_4D_test.jl")
    end

    # include("Z_q_4D_test.jl")
end