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
        "Home"                                      => "index.md",
        "Runner"                                    => "lib/Runner/Runner.md",
        "Quadrature"                                => Any[
            "Quadrature"                            => "lib/Quadrature/Quadrature.md",
            "QuadratureRuleSpec"                              => "lib/Quadrature/QuadratureRuleSpec.md",
            "QuadratureNodes"                       => "lib/Quadrature/QuadratureNodes.md",
            "QuadratureDispatch"                    => Any[
                "QuadratureDispatch"                => "lib/Quadrature/QuadratureDispatch.md",
                "QuadratureDispatchCUDA"            => "lib/Quadrature/QuadratureDispatch/QuadratureDispatchCUDA.md",
                "QuadratureDispatchThreadedSubgrid" => "lib/Quadrature/QuadratureDispatch/QuadratureDispatchThreadedSubgrid.md",
            ],
            "BSpline"                               => "lib/Quadrature/BSpline.md",
            "Gauss"                                 => "lib/Quadrature/Gauss.md",
            "NewtonCotes"                           => "lib/Quadrature/NewtonCotes.md",
        ],
        "ErrorEstimate"                             => Any[
            "ErrorEstimate"                         => "lib/ErrorEstimate/ErrorEstimate.md",
            "AutoDerivative"                        => Any[
                "AutoDerivative"                    => "lib/ErrorEstimate/AutoDerivative/AutoDerivative.md",
                "AutoDerivativeDirect"              => Any[
                    "AutoDerivativeDirect"          => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/AutoDerivativeDirect.md",
                    "ADEnzyme"                      => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/ADEnzyme.md",
                    "ADForwardDiff"                 => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/ADForwardDiff.md",
                    "ADTaylorSeries"                => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeDirect/ADTaylorSeries.md",
                ],
                "AutoDerivativeJet"                 => Any[
                    "AutoDerivativeJet"             => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeJet/AutoDerivativeJet.md",
                    "ADEnzyme"                      => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeJet/ADEnzyme.md",
                    "ADForwardDiff"                 => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeJet/ADForwardDiff.md",
                    "ADTaylorSeries"                => "lib/ErrorEstimate/AutoDerivative/AutoDerivativeJet/ADTaylorSeries.md",
                ],
            ],
            "ErrorDispatch"                         => Any[
                "ErrorDispatch"                     => "lib/ErrorEstimate/ErrorDispatch/ErrorDispatch.md",
                "ErrorDispatchDerivative"           => "lib/ErrorEstimate/ErrorDispatch/ErrorDispatchDerivative.md",
                "ErrorDispatchRefinement"           => "lib/ErrorEstimate/ErrorDispatch/ErrorDispatchRefinement.md",
            ],
            "ErrorBSpline"                          => Any[
                "ErrorBSpline"                      => "lib/ErrorEstimate/ErrorBSpline/ErrorBSpline.md",
                "ErrorBSplineDerivative"            => "lib/ErrorEstimate/ErrorBSpline/ErrorBSplineDerivative.md",
                "ErrorBSplineRefinement"            => "lib/ErrorEstimate/ErrorBSpline/ErrorBSplineRefinement.md",
            ],
            "ErrorGauss"                            => Any[
                "ErrorGauss"                        => "lib/ErrorEstimate/ErrorGauss/ErrorGauss.md",
                "ErrorGaussDerivative"              => "lib/ErrorEstimate/ErrorGauss/ErrorGaussDerivative.md",
                "ErrorGaussRefinement"              => "lib/ErrorEstimate/ErrorGauss/ErrorGaussRefinement.md",
            ],
            "ErrorNewtonCotes"                      => Any[
                "ErrorNewtonCotes"                  => "lib/ErrorEstimate/ErrorNewtonCotes/ErrorNewtonCotes.md",
                "ErrorNewtonCotesDerivative"        => "lib/ErrorEstimate/ErrorNewtonCotes/ErrorNewtonCotesDerivative.md",
                "ErrorNewtonCotesRefinement"        => "lib/ErrorEstimate/ErrorNewtonCotes/ErrorNewtonCotesRefinement.md",
            ],
        ],
        "LeastChiSquareFit"                         => "lib/LeastChiSquareFit/LeastChiSquareFit.md",
        "Integrands"                                => "lib/Integrands/Integrands.md",
        "Documentation"                             => Any[
            "Documentation"                         => "lib/Documentation/Documentation.md",
            "PlotTools"                             => "lib/Documentation/PlotTools.md",
            "Reporter"                              => "lib/Documentation/Reporter.md",
            "DocUtils"                              => "lib/Documentation/DocUtils.md",
        ],
        "Utils"                                     => Any[
            "Utils"                                 => "lib/Utils/Utils.md",
            "AvgErrFormatter"                       => "lib/Utils/AvgErrFormatter.md",
            "JobLoggerTools"                        => "lib/Utils/JobLoggerTools.md",
            "MaranathaIO"                           => "lib/Utils/MaranathaIO.md",
            "MaranathaTOML"                         => "lib/Utils/MaranathaTOML.md",
            "QuadratureBoundarySpec"                          => "lib/Utils/QuadratureBoundarySpec.md",
            "Wizard"                                => "lib/Utils/Wizard.md",
        ]
    ],
    checkdocs = :none,
    clean = false,
    warnonly = ("strict=false" in ARGS),
    doctest = ("doctest=only" in ARGS) ? :only : true,
)

# ============================================================================
# Deploy docs for Maranatha.jl
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
        repo   = "github.com/saintbenjamin/Maranatha.jl.git",
        branch = "gh-pages",
        devbranch = "main",
        target    = "build",
        forcepush = true,
    )
end