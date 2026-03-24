using Test
using Maranatha

@testset "I/O integration test" begin
    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------
    _extract_err(e) = hasproperty(e, :total) ? getproperty(e, :total) :
                      hasproperty(e, :estimate) ? getproperty(e, :estimate) :
                      error("Unsupported error-info object: need :total or :estimate")

    function _assert_run_result_basic(rr, expected_len::Int)
        @test length(rr.avg) == expected_len
        @test length(rr.err) == expected_len
        @test all(isfinite, rr.avg)
        @test all(e -> isfinite(_extract_err(e)), rr.err)
    end

    function _result_file_path(save_path, result_string, rule, boundary, ns)
        Nstr = join(sort(ns), "_")
        return joinpath(
            save_path,
            "result_$(result_string)_$(rule)_$(boundary)_N_$(Nstr).jld2"
        )
    end

    # ------------------------------------------------------------
    # Common setup
    # ------------------------------------------------------------
    integrand(x) = sin(x)
    bounds = (0.0, pi)

    dim = 1
    rule = :gauss_p4
    boundary = :LU_EXEX
    err_method = :refinement
    fit_terms = 4
    nerr_terms = 3
    ff_shift = 0
    use_error_jet = false
    write_summary = true
    use_cuda = false
    real_type = Float64

    mktempdir() do save_path
        result_string = "io_test_1d"

        # ============================================================
        # 1) Full run -> save -> load -> fit
        # ============================================================
        @testset "save/load/fit full datapoints" begin
            ns_full = [2, 3, 4, 5, 6, 7, 8, 9]

            run_result = run_Maranatha(
                integrand,
                bounds...;
                dim = dim,
                nsamples = ns_full,
                rule = rule,
                boundary = boundary,
                err_method = err_method,
                fit_terms = fit_terms,
                nerr_terms = nerr_terms,
                ff_shift = ff_shift,
                use_error_jet = use_error_jet,
                name_prefix = result_string,
                save_path = save_path,
                write_summary = write_summary,
                use_cuda = use_cuda,
                real_type = real_type
            )

            _assert_run_result_basic(run_result, length(ns_full))

            run_result_file = _result_file_path(
                save_path, result_string, rule, boundary, ns_full
            )
            @test isfile(run_result_file)

            run_result_loaded = load_datapoint_results(run_result_file)
            _assert_run_result_basic(run_result_loaded, length(ns_full))

            @test length(run_result_loaded.avg) == length(run_result.avg)
            @test all(isapprox.(run_result_loaded.avg, run_result.avg; rtol=1e-12, atol=1e-12))

            fit_result_loaded = least_chi_square_fit(
                run_result_loaded;
                fit_func_terms = fit_terms,
                ff_shift = ff_shift,
                nerr_terms = nerr_terms
            )

            @test isfinite(fit_result_loaded.estimate)
            @test isfinite(fit_result_loaded.error_estimate)
        end

        # ============================================================
        # 2) Split runs -> merge -> load -> fit
        # ============================================================
        @testset "merge partial result files and fit" begin
            ns_part1 = [2, 3]
            ns_part2 = [4, 5, 6]
            ns_part3 = [7, 8, 9]

            run_result_part1 = run_Maranatha(
                integrand,
                bounds...;
                dim = dim,
                nsamples = ns_part1,
                rule = rule,
                boundary = boundary,
                err_method = err_method,
                fit_terms = fit_terms,
                nerr_terms = nerr_terms,
                ff_shift = ff_shift,
                use_error_jet = use_error_jet,
                name_prefix = result_string,
                save_path = save_path,
                write_summary = write_summary,
                use_cuda = use_cuda,
                real_type = real_type
            )
            _assert_run_result_basic(run_result_part1, length(ns_part1))

            run_result_part2 = run_Maranatha(
                integrand,
                bounds...;
                dim = dim,
                nsamples = ns_part2,
                rule = rule,
                boundary = boundary,
                err_method = err_method,
                fit_terms = fit_terms,
                nerr_terms = nerr_terms,
                ff_shift = ff_shift,
                use_error_jet = use_error_jet,
                name_prefix = result_string,
                save_path = save_path,
                write_summary = write_summary,
                use_cuda = use_cuda,
                real_type = real_type
            )
            _assert_run_result_basic(run_result_part2, length(ns_part2))

            run_result_part3 = run_Maranatha(
                integrand,
                bounds...;
                dim = dim,
                nsamples = ns_part3,
                rule = rule,
                boundary = boundary,
                err_method = err_method,
                fit_terms = fit_terms,
                nerr_terms = nerr_terms,
                ff_shift = ff_shift,
                use_error_jet = use_error_jet,
                name_prefix = result_string,
                save_path = save_path,
                write_summary = write_summary,
                use_cuda = use_cuda,
                real_type = real_type
            )
            _assert_run_result_basic(run_result_part3, length(ns_part3))

            part1_file = _result_file_path(save_path, result_string, rule, boundary, ns_part1)
            part2_file = _result_file_path(save_path, result_string, rule, boundary, ns_part2)
            part3_file = _result_file_path(save_path, result_string, rule, boundary, ns_part3)

            @test isfile(part1_file)
            @test isfile(part2_file)
            @test isfile(part3_file)

            merged_file = merge_datapoint_result_files(
                part1_file,
                part2_file,
                part3_file;
                write_summary = true,
                output_dir = save_path,
                name_prefix = result_string,
                name_suffix = "merged",
            )

            @test isfile(merged_file)

            run_result_merged = load_datapoint_results(merged_file)
            _assert_run_result_basic(run_result_merged, 8)

            fit_result_merged = least_chi_square_fit(
                run_result_merged;
                fit_func_terms = fit_terms,
                ff_shift = ff_shift,
                nerr_terms = nerr_terms
            )

            @test isfinite(fit_result_merged.estimate)
            @test isfinite(fit_result_merged.error_estimate)
        end

        # ============================================================
        # 3) Full file -> drop some nsamples -> load -> fit
        # ============================================================
        @testset "drop nsamples from saved file and fit" begin
            ns_full = [2, 3, 4, 5, 6, 7, 8, 9]
            full_file = _result_file_path(save_path, result_string, rule, boundary, ns_full)

            @test isfile(full_file)

            filtered_file = drop_nsamples_from_file(
                full_file,
                [2, 3];
                write_summary = true,
                output_dir = save_path,
                name_prefix = result_string,
                name_suffix = "filtered"
            )

            @test isfile(filtered_file)

            run_result_filtered = load_datapoint_results(filtered_file)
            _assert_run_result_basic(run_result_filtered, 6)

            fit_result_filtered = least_chi_square_fit(
                run_result_filtered;
                fit_func_terms = fit_terms,
                ff_shift = ff_shift,
                nerr_terms = nerr_terms
            )

            @test isfinite(fit_result_filtered.estimate)
            @test isfinite(fit_result_filtered.error_estimate)
        end
    end
end