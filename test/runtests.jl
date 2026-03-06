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
using Maranatha

# ----------------------------------------------------------------------------
# Optional plotting switch:
#   MARANATHA_PLOT=1  -> enable plots
#   (default)         -> compute everything but do not call plot_convergence_result
# ----------------------------------------------------------------------------
const DO_PLOT = get(ENV, "MARANATHA_PLOT", "0") == "1"

function maybe_plot(
    a, 
    b, 
    tag::AbstractString, 
    h, 
    avg, 
    err, 
    fit; 
    rule::Symbol, 
    boundary::Symbol, 
    save_file::Bool
)
    if DO_PLOT
        plot_convergence_result(
            a, 
            b, 
            tag, 
            h, 
            avg, 
            err, 
            fit; 
            rule=rule, 
            boundary=boundary,
            save_file=save_file)
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
    @test all(e -> isfinite(e.total), res.err)
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

@testset "Maranatha.jl Quadrature Suite" begin
    announce("Maranatha.jl Quadrature Suite")
#= 
    @testset "Canonical integrands (multi-dim, multi-rule)" begin
        announce("Canonical integrands (multi-dim, multi-rule)")
        include("canonical_1D_newton_test.jl")
        include("canonical_2D_newton_test.jl")
        include("canonical_3D_newton_test.jl")
        include("canonical_4D_newton_test.jl")
        include("canonical_1D_gauss_test.jl")
        include("canonical_2D_gauss_test.jl")
        include("canonical_3D_gauss_test.jl")
        include("canonical_4D_gauss_test.jl")
        include("canonical_1D_bspline_test.jl")
        include("canonical_2D_bspline_test.jl")
        include("canonical_3D_bspline_test.jl")
        include("canonical_4D_bspline_test.jl")
    end

    include("pedagogical_visualization_1D_test.jl") =#

    # include("complicated_1D_F0000_test.jl")
    include("complicated_4D_Z_q_test.jl")
end