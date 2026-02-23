# __precompile__(false)

module Maranatha

# === Include numerical integration rules ===
include("rules/Simpson13Rule.jl")
include("rules/Simpson38Rule.jl")
include("rules/BodeRule.jl")

using .Simpson13Rule
using .Simpson38Rule
using .BodeRule

include("rules/Simpson13Rule_MinOpen_MaxOpen.jl")
include("rules/Simpson38Rule_MinOpen_MaxOpen.jl")
include("rules/BodeRule_MinOpen_MaxOpen.jl")

using .Simpson13Rule_MinOpen_MaxOpen
using .Simpson38Rule_MinOpen_MaxOpen
using .BodeRule_MinOpen_MaxOpen

include("rules/Integrate.jl")

using .Integrate

# === Include error estimation and fitting tools ===
include("error/ErrorEstimator.jl")
include("error/RichardsonError.jl")
include("fit/AvgErrFormatter.jl")
include("fit/FitConvergence.jl")

using .ErrorEstimator
using .RichardsonError
using .AvgErrFormatter
using .FitConvergence

include("controller/Runner.jl")

using .Runner

end  # module Maranatha