# ============================================================================
# src/Documentation/Reporter/datapoints/write_convergence_summary_datapoints.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    write_convergence_summary_datapoints(
        result;
        name::String = "Maranatha",
        h_power::Real = 1,
        xscale::Symbol = :linear,
        yscale::Symbol = :linear,
        out_dir::String = ".",
        format::Symbol = :tex,
        save_file::Bool = true,
    ) -> String

Generate a datapoints-only convergence summary report in
[``\\LaTeX``](https://www.latex-project.org/) or Markdown format directly from
a structured quadrature result object.

# Function description

This routine builds a formatted summary of raw quadrature datapoints without
requiring a fitted extrapolation model.

It allows users to generate a datapoints-only summary directly from a stored
or freshly computed Maranatha result without manually unpacking arrays.

It is intended for pre-fit inspection or archival reporting when the user wants
to document:

- the integration interval,
- the quadrature rule and boundary configuration,
- the horizontal plotting convention ``x = h^p``,
- the filtered quadrature estimates and their associated uncertainties.

Unlike [`write_convergence_summary`](@ref), this function does **not** include
fit parameters, extrapolated values, or goodness-of-fit statistics.

# Arguments

- `result`:
  Result object exposing fields such as `a`, `b`, `h`, `avg`, `err`, `rule`,
  and `boundary`.

  The summary is generated from:

  - `result.a`, `result.b` as the integration-domain description,
  - `result.h` as the scalar step-size sequence,
  - `result.avg` as the corresponding quadrature estimates,
  - `result.err` as the error objects containing pointwise uncertainties,
  - `result.rule` and `result.boundary` as reporting labels.

  In rectangular-domain workflows, `result.h` is expected to be the scalarized
  step-size sequence used for plotting and reporting, while any original
  per-axis step data remain outside this helper.

  Each entry of `result.err` is expected to provide either:

  - a `.total` field, as in the residual-based error-estimation workflow, or
  - an `.estimate` field, as in the refinement-based error-estimation workflow.

  The reporting routine converts each entry into a nonnegative scalar
  uncertainty through an internal extractor.

# Keyword arguments

- `name::String = "Maranatha"`:
  Identifier used in the generated report and output filename.
- `h_power::Real = 1`:
  Power used to define the horizontal coordinate ``x = h^{p}``.
- `xscale::Symbol = :linear`:
  Horizontal axis scale (`:linear` or `:log`).
- `yscale::Symbol = :linear`:
  Vertical axis scale (`:linear` or `:log`).
- `out_dir::String = "."`:
  Output directory used when writing the summary file.
- `format::Symbol = :tex`:
  Output format (`:tex` or `:md`).
- `save_file::Bool = true`:
  If `true`, write the generated text to disk.

# Returns

- `String`:
  The generated summary text.

# Errors

- Throws via [`JobLoggerTools.error_benji`](@ref) if the input lengths are inconsistent.
- Throws if unsupported axis-scale keywords are supplied.
- Throws if no valid datapoints remain after filtering.

# Notes

- Non-finite datapoints are removed automatically before reporting.
- Additional positivity filters are applied when `xscale == :log` or `yscale == :log`.
- The `name` argument may be a simple identifier or a file path; file-output
  basenames are sanitized internally via [`DocUtils._split_report_name`](@ref).
- Datapoints are ordered from coarse to fine resolution (largest `h` first) in
  the final report.
- This routine accepts both residual-based and refinement-based error-info
  objects, provided that each entry exposes either `.total` or `.estimate`.
- For rectangular-domain workflows, the generated summary is based on the
  scalarized step-size sequence `result.h`.
- When any of `a`, `b`, `rule`, or `boundary` is axis-wise, the run
  configuration table expands into one row per axis.
- Saved basenames encode rule/boundary metadata through the axis-aware token
  produced by [`DocUtils._rule_boundary_filename_token`](@ref).
"""
function write_convergence_summary_datapoints(
    result;
    name::String = "Maranatha",
    h_power::Real = 1,
    xscale::Symbol = :linear,
    yscale::Symbol = :linear,
    out_dir::String = ".",
    format::Symbol = :tex,
    save_file::Bool = true,
)
    a = result.a
    b = result.b
    hs = Vector{Float64}(result.h)
    estimates = Vector{Float64}(result.avg)
    errors = result.err
    rule = result.rule
    boundary = result.boundary

    n = length(hs)
    if length(estimates) != n || length(errors) != n
        JobLoggerTools.error_benji("Input length mismatch.")
    end

    (xscale == :linear || xscale == :log) || JobLoggerTools.error_benji(
        "Unsupported xscale=$xscale (expected :linear or :log)"
    )
    (yscale == :linear || yscale == :log) || JobLoggerTools.error_benji(
        "Unsupported yscale=$yscale (expected :linear or :log)"
    )

    @inline function _extract_error_total(e)
        if hasproperty(e, :total)
            return float(e.total)
        elseif hasproperty(e, :estimate)
            return abs(float(e.estimate))
        else
            JobLoggerTools.error_benji(
                "Unsupported error-info structure (need :total or :estimate)."
            )
        end
    end

    display_name, file_name = DocUtils._split_report_name(name)

    hxp = Float64.(hs) .^ float(h_power)
    errp = [_extract_error_total(e) for e in errors]

    mask = isfinite.(hs) .& isfinite.(hxp) .& isfinite.(estimates) .& isfinite.(errp)

    if xscale == :log
        mask .&= hxp .> 0
    end
    if yscale == :log
        mask .&= estimates .> 0
    end

    hsp  = hs[mask]
    hxp  = hxp[mask]
    estp = estimates[mask]
    errp = errp[mask]

    isempty(hsp) && JobLoggerTools.error_benji("No valid datapoints remain after filtering.")

    perm = sortperm(hsp; rev=true)
    hsp  = hsp[perm]
    hxp  = hxp[perm]
    estp = estp[perm]
    errp = errp[perm]

    summary_basename = _build_convergence_summary_datapoints_basename(
        file_name, a, b, rule, boundary, h_power, xscale, yscale
    )

    if format == :tex
        text = _build_convergence_summary_datapoints_tex(
            a, b, display_name, hsp, hxp, estp, errp;
            h_power = h_power,
            xscale = xscale,
            yscale = yscale,
            rule = rule,
            boundary = boundary,
        )
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).tex")
            open(outfile, "w") do io
                write(io, text)
            end
        end
        return text

    elseif format == :md
        text = _build_convergence_summary_datapoints_md(
            a, b, display_name, hsp, hxp, estp, errp;
            h_power = h_power,
            xscale = xscale,
            yscale = yscale,
            rule = rule,
            boundary = boundary,
        )
        if save_file
            mkpath(out_dir)
            outfile = joinpath(out_dir, "$(summary_basename).md")
            open(outfile, "w") do io
                write(io, text)
            end
        end
        return text
    else
        JobLoggerTools.error_benji("Unsupported format=$(format). Use :tex or :md.")
    end
end
