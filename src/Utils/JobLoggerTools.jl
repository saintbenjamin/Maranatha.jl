# ============================================================================
# src/Utils/JobLoggerTools.jl
#
# Shared module mirrored between Maranatha.jl and Deborah.jl.
# Historical origin: Deborah.jl
#
# The two copies are maintained as mirrored implementations and may evolve
# independently for short periods. Changes made here should be reviewed
# against the corresponding Deborah.jl file and synchronized as appropriate.
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module JobLoggerTools

Lightweight logging and assertion helpers shared across `Maranatha.jl`.

# Module description
`Maranatha.Utils.JobLoggerTools` collects small convenience wrappers for
timestamped logging, staged progress messages, warnings, assertions, and
fatal-error reporting.

# Main entry points
- [`println_benji`](@ref)
- [`warn_benji`](@ref)
- [`error_benji`](@ref)
- [`assert_benji`](@ref)
"""
module JobLoggerTools

import ..Printf: @sprintf
import ..Dates

"""
    @logtime_benji(
        jobid_expr, 
        expr
    ) -> Any

Evaluate an expression, measure its runtime, and log timing information.

# Function description
This macro executes `expr` under `@timed`, computes elapsed time, inspects GC
allocation statistics before and after evaluation, and prints a timestamped log
message.

If `jobid_expr` evaluates to a non-`nothing` value, it is included as a log
prefix. The macro returns the value produced by `expr`.

# Arguments
- `jobid_expr`: Optional job identifier expression.
- `expr`: Julia expression to evaluate and time.

# Returns
- The evaluated result of `expr`.

# Errors
- Propagates any exception thrown while evaluating `expr`.

# Notes
- Allocation reporting is based on `Base.gc_num()` / `Base.GC_Diff`.
- The log message is printed to standard output.
"""
macro logtime_benji(
    jobid_expr, 
    expr
)
    esc_jobid = esc(jobid_expr)
    quote
        _jobid = $esc_jobid
        _prefix = (_jobid === nothing) ? "" : "[$_jobid] "

        _gc_before = Base.gc_num()
        res = @timed $(esc(expr))
        _gc_after = Base.gc_num()

        diff = Base.GC_Diff(_gc_after, _gc_before)
        bytes = diff.allocd / 1024^2

        parts = String[]
        push!(parts, @sprintf("%.6f seconds", res.time))
        if bytes > 0
            push!(parts, @sprintf("(%.3f MiB allocated)", bytes))
        end

        msg = join(parts, " ")
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.sss")
        println(_prefix * "[$timestamp] " * msg)

        res.value
    end
end

"""
    println_benji(
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a timestamped log message with an optional job ID prefix.

# Function description
This helper formats the current timestamp, prepends an optional job ID, writes
the resulting message to `stdout`, and flushes the output stream immediately.

# Arguments
- `msg::AbstractString`: Message to print.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Output is written to `stdout`.
- Flushing is performed after every call.
"""
function println_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.sss")
    prefix = isnothing(jobid) ? "" : "[$jobid] "
    msg_full = "$prefix[$timestamp] $msg"
    println(stdout, msg_full)
    flush(stdout)
end

"""
    log_stage_benji(
        title::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a high-level stage delimiter block.

# Function description
This helper emits a visually separated logging block consisting of blank lines,
a repeated `=` delimiter, and a titled stage header.

# Arguments
- `title::String`: Stage title to display.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Intended for major stage boundaries in longer logs.
"""
function log_stage_benji(
    title::String, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    println_benji("", jobid)
    println_benji(repeat("=", 50), jobid)
    println_benji(">>> " * title,  jobid)
    println_benji(repeat("=", 50), jobid)
    println_benji("", jobid)
end

"""
    log_stage_sub1_benji(
        title::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a substage delimiter block.

# Function description
This helper emits a visually separated logging block consisting of blank lines,
a repeated `-` delimiter, and a titled substage header.

# Arguments
- `title::String`: Substage title to display.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Intended for subsection-level boundaries inside a larger stage.
"""
function log_stage_sub1_benji(
    title::String, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    println_benji("", jobid)
    println_benji(repeat("-", 50), jobid)
    println_benji(">> " * title,  jobid)
    println_benji(repeat("-", 50), jobid)
    println_benji("", jobid)
end

"""
    error_benji(
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Log an error message, flush output streams, and throw an exception.

# Function description
This helper prints `msg` through [`println_benji`](@ref), flushes both `stdout`
and `stderr`, and then raises `error(msg)`.

# Arguments
- `msg::AbstractString`: Error message.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- This function does not return normally.

# Errors
- Always throws an exception via `error(msg)`.

# Notes
- The printed message and the thrown message are the same.
"""
function error_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)
    println_benji(msg, jobid)
    flush(stdout)
    flush(stderr)
    error(msg)
end

"""
    warn_benji(
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a timestamped warning message.

# Function description
This helper prefixes the message with `[WARNING]` and forwards it to
[`println_benji`](@ref).

# Arguments
- `msg::AbstractString`: Warning message.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Output is written to `stdout`.
"""
function warn_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    full_msg = "[WARNING] $msg"
    println_benji(full_msg, jobid)
end

"""
    assert_benji(
        cond::Bool, 
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Assert that `cond` is true, otherwise log and throw.

# Function description
If `cond` is false, this helper calls [`error_benji`](@ref) with an assertion
failure prefix. If `cond` is true, it returns normally.

# Arguments
- `cond::Bool`: Condition to assert.
- `msg::AbstractString`: Assertion-failure message.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing` when the assertion passes.

# Errors
- Throws via [`error_benji`](@ref) if `cond` is false.

# Notes
- The emitted message is prefixed with `[ASSERTION FAILED]`.
"""
function assert_benji(
    cond::Bool, 
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    if !cond
        error_benji("[ASSERTION FAILED] $msg", jobid)
    end
end

"""
    info_benji(
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a timestamped informational message.

# Function description
This helper prefixes the message with `[INFO]` and forwards it to
[`println_benji`](@ref).

# Arguments
- `msg::AbstractString`: Informational message.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Output is written to `stdout`.
"""
function info_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    full_msg = "[INFO] $msg"
    println_benji(full_msg, jobid)
end

"""
    debug_benji(
        msg::AbstractString, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a timestamped debug message.

# Function description
This helper prefixes the message with `[DEBUG]` and forwards it to
[`println_benji`](@ref).

# Arguments
- `msg::AbstractString`: Debug message.
- `jobid::Union{Nothing, String}`: Optional job identifier.

# Returns
- `Nothing`.

# Errors
- No explicit validation is performed.

# Notes
- Output is written to `stdout`.
"""
function debug_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    full_msg = "[DEBUG] $msg"
    println_benji(full_msg, jobid)
end

end  # module JobLoggerTools
