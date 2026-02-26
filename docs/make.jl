# ============================================================================
# docs/make.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

ENV["DOCUMENTER_DEBUG"] = "true"

using Documenter, Maranatha

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

linkcheck_ignore = [
    # We'll ignore links that point to GitHub's edit pages, as they redirect to the
    # login screen and cause a warning:
    r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)",
    "https://nvd.nist.gov/vuln/detail/CVE-2018-16487",
    # We'll ignore the links to Documenter tags in CHANGELOG.md, since when you tag
    # a release, the release link does not exist yet, and this will cause the linkcheck
    # CI job to fail on the PR that tags a new release.
    r"https://github.com/JuliaDocs/Documenter.jl/releases/tag/v1.\d+.\d+",
]
# Extra ones we ignore only on CI.
if get(ENV, "GITHUB_ACTIONS", nothing) == "true"
    # It seems that CTAN blocks GitHub Actions?
    push!(linkcheck_ignore, "https://ctan.org/pkg/minted")
end

makedocs(
    modules = [Maranatha],
    format = if "pdf" in ARGS
        Documenter.LaTeX(
            platform = "native"
        )
    else
        Documenter.HTML(
            # sidebar_sitename = false,
            edit_link = nothing
        )
    end,
    build = ("pdf" in ARGS) ? "build-pdf" : "build",
    debug = ("pdf" in ARGS),
    authors = "Benjamin J. Choi",
    sitename = "Maranatha.jl",
    linkcheck = "linkcheck" in ARGS,
    linkcheck_ignore = linkcheck_ignore,
    pages = [
        "Home" => "index.md",
        "Reference" => Any[
            "Runner"                          => "lib/Runner.md",
            "Integrands"                      => "lib/Integrands.md",
            "Integrate"                       => "lib/Integrate.md",
            "ErrorEstimator"                  => "lib/ErrorEstimator.md",
            "FitConvergence"                  => "lib/FitConvergence.md",
            "PlotTools"                       => "lib/PlotTools.md",
            "AvgErrFormatter"                 => "lib/AvgErrFormatter.md",
            "JobLoggerTools"                  => "lib/JobLoggerTools.md",
            "Legacy" => Any[
                "Integrate (Legacy)"              => "lib/Integrate_legacy.md",
                "ErrorEstimator (Legacy)"         => "lib/ErrorEstimator_legacy.md",
                "RichardsonError"                 => "lib/RichardsonError.md",
                "Simpson13Rule"                   => "lib/Simpson13Rule.md",
                "Simpson13Rule_MinOpen_MaxOpen"   => "lib/Simpson13Rule_MinOpen_MaxOpen.md",
                "Simpson38Rule"                   => "lib/Simpson38Rule.md",
                "Simpson38Rule_MinOpen_MaxOpen"   => "lib/Simpson38Rule_MinOpen_MaxOpen.md",
                "BodeRule"                        => "lib/BodeRule.md",
                "BodeRule_MinOpen_MaxOpen"        => "lib/BodeRule_MinOpen_MaxOpen.md",
                "F0000GammaEminus1"               => "lib/F0000GammaEminus1.md",
                "F0000Preset"                     => "lib/F0000Preset.md",
                "Z_q"                             => "lib/Z_q.md",
            ]
        ]
    ],
    checkdocs = :none,
    clean = false,
    warnonly = ("strict=false" in ARGS),
    doctest = ("doctest=only" in ARGS) ? :only : true,
)

# ============================================================================
# Deploy docs
# ============================================================================

if "pdf" in ARGS
    # Move only the generated PDF into a dedicated commit directory
    pdf_commit_dir = joinpath(@__DIR__, "build-pdf", "commit")
    mkpath(pdf_commit_dir)

    for f in readdir(joinpath(@__DIR__, "build-pdf"))
        if endswith(f, ".pdf")
            mv(
                joinpath(@__DIR__, "build-pdf", f),
                joinpath(pdf_commit_dir, f),
                force = true,
            )
        end
    end

    deploydocs(
        repo   = "github.com/saintbenjamin/Maranatha.jl.git",
        target = "build-pdf/commit",
        branch = "gh-pages-pdf",
        devbranch = "main",
        forcepush = true,
    )

else
    deploydocs(
        repo   = "github.com/saintbenjamin/Maranatha.git",
        branch = "gh-pages",
        devbranch = "main",
        target    = "build",
        forcepush = true,
    )
end