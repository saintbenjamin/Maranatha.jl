# ============================================================================
# src/log/JobLoggerTools.jl (Benji: taken from Deborah.Sarah)
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module JobLoggerTools

import Printf: @sprintf
import Dates

"""
    @logtime_benji(
        jobid_expr, 
        expr
    ) -> Any

Macro that times and logs the execution of an expression, with optional GC allocation info.

# Arguments
- `jobid_expr`: Optional job ID (can be `nothing`).
- `expr`: Any Julia expression to execute and time.

Logs elapsed time and memory usage with a timestamp. Returns the value of `expr`.
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

Print a timestamped log message with optional job ID.

# Arguments
- `msg`: The message to print.
- `jobid`: Optional job identifier to prepend.

Output is printed to `stdout`, immediately flushed.
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

Print a high-level stage delimiter with a title, surrounded by `=` lines.

Useful for separating major processing stages in logs.
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

Print a substage delimiter with a title, surrounded by `-` lines.

Used for visually marking sub-sections in logs.
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

Print an error message with timestamp and job ID (if given), then throw an error with the same message.

Also flushes `stdout` and `stderr` after printing.
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

Print a timestamped warning message with optional job ID.

The message is prefixed with `[WARNING]`, and flushed to `stdout`.

# Arguments
- `msg`: Warning message to print.
- `jobid`: Optional job identifier.
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

Assert that `cond` is true. If false, log an error with timestamp and job ID, then throw.

# Arguments
- `cond`: Boolean condition to assert.
- `msg`: Message to print if assertion fails.
- `jobid`: Optional job identifier.
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

Print a timestamped informational message with optional job ID.

The message is prefixed with `[INFO]`, and flushed to `stdout`.

# Arguments
- `msg`: Informational message to print.
- `jobid`: Optional job identifier.
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

Print a timestamped debug message with optional job ID.

The message is prefixed with `[DEBUG]`, and flushed to `stdout`.

# Arguments
- `msg`: Debug message to print.
- `jobid`: Optional job identifier.
"""
function debug_benji(
    msg::AbstractString, 
    jobid::Union{Nothing, String}=nothing
)::Nothing
    full_msg = "[DEBUG] $msg"
    println_benji(full_msg, jobid)
end

end  # module JobLoggerTools