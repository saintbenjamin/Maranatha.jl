# ============================================================================
# src/Utils/Utils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Utils

Unified utility bundle used across `Maranatha.jl`.

# Overview

`Maranatha.Utils` serves as a lightweight aggregation layer for reusable helper
submodules shared throughout the codebase.

Rather than collecting unrelated helpers in one large file, this module groups
utilities into dedicated submodules and re-exposes them under a common
namespace.

Currently included:

- [`Maranatha.Utils.JobLoggerTools`](@ref)
- [`Maranatha.Utils.AvgErrFormatter`](@ref)
- [`Maranatha.Utils.QuadratureBoundarySpec`](@ref)
- [`Maranatha.Utils.MaranathaIO`](@ref)
- [`Maranatha.Utils.MaranathaTOML`](@ref)
- [`Maranatha.Utils.Wizard`](@ref)

# Design philosophy

- Keep utilities modular and dependency-light.
- Avoid monolithic helper files.
- Provide a stable shared namespace.
- Allow future extension by adding submodules without breaking existing interfaces.

`Maranatha.Utils` is intentionally infrastructural and does not implement the
main numerical algorithms of the package.

# Notes

- This module primarily exists to group logging, formatting, I/O, TOML, and
  helper workflow utilities under one namespace.
- Higher-level modules may import individual submodules or access them through
  `Maranatha.Utils.*`.
"""
module Utils

import ..Printf
import ..TOML
import ..DoubleFloats
import ..LinearAlgebra
import Dates
import JLD2

include("JobLoggerTools.jl")
include("AvgErrFormatter.jl")
include("MaranathaIO/MaranathaIO.jl")
include("QuadratureBoundarySpec.jl")
include("MaranathaTOML/MaranathaTOML.jl")
include("Wizard.jl")

using .JobLoggerTools
using .AvgErrFormatter
using .MaranathaIO
using .QuadratureBoundarySpec
using .MaranathaTOML
using .Wizard

end  # module Utils
