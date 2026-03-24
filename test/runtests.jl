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
using DoubleFloats

# ----------------------------------------------------------------------------
# Optional plotting switch:
#   MARANATHA_PLOT=1  -> enable plots
#   (default)         -> compute everything but do not call plot_convergence_result
# ----------------------------------------------------------------------------
const DO_PLOT = get(ENV, "MARANATHA_PLOT", "0") == "1"

# ----------------------------------------------------------------------------
# Smoke checks: basic sanity without assuming exact numbers
# ----------------------------------------------------------------------------
function _extract_err_total(e)
    if hasproperty(e, :total)
        return e.total
    elseif hasproperty(e, :estimate)
        return e.estimate
    else
        error("Unsupported error entry: expected field :total or :estimate")
    end
end

function assert_result_sane(res)
    @test length(res.h) == length(res.avg) == length(res.err)
    @test all(isfinite, res.h)
    @test all(isfinite, res.avg)
    @test all(e -> isfinite(_extract_err_total(e)), res.err)
    @test all(>(0.0), res.h)  # step sizes should be positive
    @test issorted(res.h; rev=true)
    return nothing
end

function include_with_announce(path::AbstractString)
    println()
    println("▶ running: ", path)
    t0 = time_ns()
    include(path)
    dt = (time_ns() - t0) / 1e9
    println("✓ done: ", path, " (", round(dt; digits=2), " s)")
    println()
    return nothing
end

@testset "Maranatha.jl Quadrature Suite" begin
    @testset "Canonical integrands (multi-dim, multi-rule)" begin
        include_with_announce("canonical/1D_newton_test.jl")
        include_with_announce("canonical/2D_newton_test.jl")
        # include_with_announce("canonical/3D_newton_test.jl")
        # include_with_announce("canonical/4D_newton_test.jl")
        include_with_announce("canonical/1D_gauss_test.jl")
        include_with_announce("canonical/2D_gauss_test.jl")
        include_with_announce("canonical/3D_gauss_test.jl")
        include_with_announce("canonical/4D_gauss_test.jl")
        include_with_announce("canonical/1D_bspline_test.jl")
        include_with_announce("canonical/2D_bspline_test.jl")
        include_with_announce("canonical/3D_bspline_test.jl")
        include_with_announce("canonical/4D_bspline_test.jl")
    end
    include_with_announce("io/1D_gauss_io_test.jl")
    include_with_announce("complicated/1D_F0000_test.jl")
    include_with_announce("regression/axiswise_boundary_derivative_test.jl")
    include_with_announce("regression/axiswise_rule_test.jl")
    include_with_announce("pedagogical/visualization_1D_test.jl")
end