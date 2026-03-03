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
submodules that are shared throughout the `Maranatha.jl` codebase.

Rather than placing miscellaneous helper functions in a single large file,
this module groups logically separated utilities into dedicated submodules
and re-exposes them under a common namespace.

Currently included:

- [`Maranatha.Utils.JobLoggerTools`](@ref)  
  Timestamped logging utilities, structured stage delimiters, assertion
  helpers, and the [`Maranatha.Utils.JobLoggerTools.@logtime_benji`](@ref) macro for execution timing with
  allocation reporting.

- [`Maranatha.Utils.AvgErrFormatter`](@ref)  
  Compact formatting utilities for central values and uncertainties,
  producing parenthetical notation such as `"1.23(45)"`, suitable for
  numerical analysis output and publication-style reporting.

# Design Philosophy

- Keep utilities **modular and dependency-light**.
- Avoid monolithic helper files.
- Provide a stable namespace (`Maranatha.Utils.*`) for shared infrastructure.
- Allow future extensions by adding new submodules without breaking
  existing interfaces.

`Maranatha.Utils` is intentionally infrastructural and contains no numerical
algorithms. Its purpose is to support logging, formatting, and other
cross-cutting concerns used by higher-level modules such as
[`Maranatha.Quadrature`](@ref), 
[`Maranatha.ErrorEstimate`](@ref), 
and [`Maranatha.LeastChiSquareFit`](@ref).

# Example

```julia
using Maranatha.Utils

Utils.JobLoggerTools.log_stage_benji("Starting computation")

result = Utils.JobLoggerTools.@logtime_benji(nothing, sum(rand(10^6)))

formatted = Utils.JobLoggerTools.avgerr_e2d_from_float(1.234, 0.056)
Utils.JobLoggerTools.println_benji("Fit result = \$formatted")
```

Additional submodules may be introduced over time as the project evolves.
"""
module Utils

include("JobLoggerTools.jl")
include("AvgErrFormatter.jl")

using .JobLoggerTools
using .AvgErrFormatter

end  # module Utils