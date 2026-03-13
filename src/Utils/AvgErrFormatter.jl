# ============================================================================
# src/Utils/AvgErrFormatter.jl
#
# Shared module mirrored between Maranatha.jl and Deborah.jl.
# Historical origin: Deborah.jl/src/Sarah/AvgErrFormatter.jl
#
# This file is maintained as a shared counterpart, not as a permanently
# authoritative source. Changes made here should be reviewed against the
# corresponding Deborah.jl file and synchronized as appropriate.
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module AvgErrFormatter

import ..Printf: @sprintf, Printf

"""
    round_sig(
        x::Float64, 
        sig::Int = 2
    ) -> Float64

Round a floating-point number to a specified number of significant digits.

# Function description
This helper rounds `x` so that the result retains `sig` significant digits.

It is mainly used as an internal formatting utility when compact uncertainty
strings need magnitude-aware rounding.

# Arguments
- `x::Float64`: Number to round.
- `sig::Int`: Number of significant digits.

# Returns
- `Float64`: Rounded value.

# Errors
- No explicit validation is performed.
- Domain issues may occur if `x == 0.0` because the implementation uses `log10(abs(x))`.

# Notes
- This helper is intended for formatting workflows rather than general-purpose
  numerical rounding.
"""
function round_sig(
    x::Float64, 
    sig::Int=2
)::Float64
    return round(x, digits = sig - Int(floor(log10(abs(x)))) - 1)
end

"""
    _avgerr_fallback_string(
        cen::Real,
        err::Real
    ) -> String

Construct a fallback string for a central value and its uncertainty.

# Function description
This helper returns a fully explicit scientific-notation representation of the
form

    "%.14e(%.14e)"

when the compact parenthetical formatting path fails.

It is intended as a last-resort output path so that numerical information is
preserved even if compact formatting cannot be completed safely.

# Arguments
- `cen::Real`: Central value.
- `err::Real`: Error value.

# Returns
- `String`: Fallback string containing both values in scientific notation.

# Errors
- No explicit errors are thrown here.

# Notes
- This helper is used internally by [`avgerr_e2d`](@ref) and
  [`_safe_avgerr_return`](@ref).
"""
@inline function _avgerr_fallback_string(
    cen::Real,
    err::Real
)::String
    return Printf.format(Printf.Format("%.14e(%.14e)"), Float64(cen), Float64(err))
end

"""
    _safe_avgerr_return(
        fmt::AbstractString,
        args...;
        cen::Real,
        err::Real
    ) -> String

Safely construct a formatted uncertainty string using a dynamic `Printf` format.

# Function description
This helper attempts to build a formatted uncertainty string using the supplied
dynamic format string and arguments. If formatting fails, it falls back to
[`_avgerr_fallback_string`](@ref).

# Arguments
- `fmt::AbstractString`: Dynamic `Printf` format string.
- `args...`: Positional arguments forwarded to the formatter.
- `cen::Real`: Central value used by the fallback path.
- `err::Real`: Error value used by the fallback path.

# Returns
- `String`: Formatted uncertainty string, or a fallback scientific-notation
  string if formatting fails.

# Errors
- No exception is propagated from the formatting attempt itself; failures are
  absorbed and redirected to the fallback formatter.

# Notes
- This helper protects only the final string-construction step.
"""
@inline function _safe_avgerr_return(
    fmt::AbstractString,
    args...;
    cen::Real,
    err::Real
)::String
    try
        return Printf.format(Printf.Format(fmt), args...)
    catch
        return _avgerr_fallback_string(cen, err)
    end
end

"""
    avgerr_e2d(
        censtr::String,
        errstr::String
    ) -> String

Format a central value and uncertainty into compact parenthetical notation.

# Function description
This routine converts a central value string and an error string into a compact
uncertainty representation such as:

- `"1.23(45)"`
- `"2.00000000000000(39)"`
- `"1.2(3) *"`

The formatting logic is magnitude-dependent and attempts to keep the output
short while preserving the numerical meaning of the uncertainty.

If the compact formatting path fails at any stage, the routine falls back to an
explicit scientific-notation representation so that the numerical information is
not lost.

# Arguments
- `censtr::String`: Central value string, typically in scientific notation.
- `errstr::String`: Error value string, typically in scientific notation.

# Returns
- `String`: Compact parenthetical uncertainty string, or a fallback scientific
  representation if compact formatting fails.

# Errors
- This routine is intentionally failure-tolerant and avoids throwing in normal
  formatting failures.
- If even fallback parsing fails, a minimal string of the form `censtr(errstr)`
  is returned.

# Notes
- Non-finite values and negative errors trigger fallback behavior.
- The implementation assumes exponent-style input compatible with strings
  produced by `@sprintf("%.14e", x)`.
"""
function avgerr_e2d(
    censtr::String,
    errstr::String
)::String
    try
        cen = parse(Float64, censtr)
        err = parse(Float64, errstr)

        # Additional hard safety: non-finite or negative error
        if !isfinite(cen) || !isfinite(err) || err < 0
            return _avgerr_fallback_string(cen, err)
        end

        # Special case: error much bigger than central value
        if err ≥ 1.0 && abs(cen) < 0.1 * err
            expo = floor(Int, log10(abs(err)))
            cen_mantissa = cen / 10.0^expo
            err_scaled = round_sig(err / 10.0^expo, 2)
            err_2digit = err_scaled < 10 ? round(Int, err_scaled * 10) : round(Int, err_scaled)

            return _safe_avgerr_return(
                "%.1fe+%02d(%02d) *",
                cen_mantissa, expo, err_2digit;
                cen=cen, err=err
            )
        end

        if err < 1.0
            digit0 = parse(Int, split(errstr, "-")[2]) - 1
            digit = digit0 + 2

            cen_round_str = @sprintf("%.*f", digit, cen)
            err_round = round_sig(err)
            err_round_int = Int(round(err_round * 10^digit))

            if abs(cen) > err
                return _safe_avgerr_return(
                    "%s(%d)",
                    cen_round_str, err_round_int;
                    cen=cen, err=err
                )
            else
                return _safe_avgerr_return(
                    "%s(%d) *",
                    cen_round_str, err_round_int;
                    cen=cen, err=err
                )
            end

        elseif 1.0 ≤ err < 10.0
            if abs(cen) > 1.0
                digit = 1
                cen_round_str = @sprintf("%.*f", digit, cen)
                err_round = round_sig(err)
                err_round_int = Int(round(err_round * 10^digit))

                if abs(cen) > err
                    return _safe_avgerr_return(
                        "%s(%d)",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                else
                    return _safe_avgerr_return(
                        "%s(%d) *",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                end
            else
                # digit0 = parse(Int, split(censtr, "-")[2])
                digit0 = parse(Int, split(censtr, "-")[end])
                digit = digit0 + 2

                cen_round_str = @sprintf("%.*f", digit0, cen)
                err_round = round_sig(err, digit)
                err_round_int = Int(round(err_round * 10^digit0))

                if abs(cen) > err
                    return _safe_avgerr_return(
                        "%s(%d)",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                else
                    return _safe_avgerr_return(
                        "%s(%d) *",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                end
            end
        else
            if abs(cen) > 1.0
                digit0 = parse(Int, split(errstr, "+")[2])
                digit = digit0 + 1

                cen_round_str = @sprintf("%.*f", digit, cen)
                cen_round = round(Int, parse(Float64, cen_round_str))
                err_round = round_sig(err, digit)
                err_round_int = Int(round(err_round))

                if abs(cen) > err
                    return _safe_avgerr_return(
                        "%d(%d)",
                        cen_round, err_round_int;
                        cen=cen, err=err
                    )
                else
                    return _safe_avgerr_return(
                        "%d(%d) *",
                        cen_round, err_round_int;
                        cen=cen, err=err
                    )
                end
            else
                # digit0 = parse(Int, split(censtr, "-")[2])
                digit0 = parse(Int, split(censtr, "-")[end])
                digit = digit0 + 2

                cen_round_str = @sprintf("%.*f", digit0, cen)
                err_round = round_sig(err, digit)
                err_round_int = Int(round(err_round * 10^digit0))

                if abs(cen) > err
                    return _safe_avgerr_return(
                        "%s(%d)",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                else
                    return _safe_avgerr_return(
                        "%s(%d) *",
                        cen_round_str, err_round_int;
                        cen=cen, err=err
                    )
                end
            end
        end

    catch
        # Last-resort fallback: even if parsing / exponent extraction / rounding fails,
        # never stop here.
        try
            cen = parse(Float64, censtr)
            err = parse(Float64, errstr)
            return _avgerr_fallback_string(cen, err)
        catch
            # Absolute last fallback if even parsing fails
            return string(censtr, "(", errstr, ")")
        end
    end
end

"""
    _group_digits_right(
        s::AbstractString;
        group::Int = 3,
        sep::AbstractString = raw"\\,"
    ) -> String

Insert digit-group separators into the integer part of a number string.

# Function description
This helper groups digits from right to left, preserving an optional leading
minus sign.

It is intended for visual formatting of integer parts such as:

- `"1234567"` -> `"1\\,234\\,567"`
- `"-1234567"` -> `"-1\\,234\\,567"`

# Arguments
- `s::AbstractString`: Integer-part string, optionally beginning with `"-"`.
- `group::Int`: Number of digits per group.
- `sep::AbstractString`: Separator inserted between groups.

# Returns
- `String`: Grouped integer-part string.

# Errors
- No explicit validation is performed.

# Notes
- This helper assumes that `s` is already an integer-like substring with no
  decimal point or exponent marker.
"""
@inline function _group_digits_right(
    s::AbstractString;
    group::Int = 3,
    sep::AbstractString = raw"\,"
)::String
    # Separate sign
    sign = startswith(s, "-") ? "-" : ""
    digits = sign == "-" ? s[2:end] : s

    n = length(digits)
    n ≤ group && return sign * digits

    r = n % group
    io = IOBuffer()

    if r != 0
        write(io, digits[1:r])
        if r < n
            write(io, sep)
        end
    end

    i = r == 0 ? 1 : r + 1
    while i ≤ n
        j = min(i + group - 1, n)
        write(io, digits[i:j])
        if j < n
            write(io, sep)
        end
        i += group
    end

    return sign * String(take!(io))
end

"""
    _group_digits_left(
        s::AbstractString;
        group::Int = 3,
        sep::AbstractString = raw"\\,"
    ) -> String

Insert digit-group separators into the fractional part of a number string.

# Function description
This helper groups digits from left to right, which is useful for visually
formatting long decimal tails in [``\\LaTeX``](https://www.latex-project.org/)-friendly output.

# Arguments
- `s::AbstractString`: Fractional-part string with no decimal point.
- `group::Int`: Number of digits per group.
- `sep::AbstractString`: Separator inserted between groups.

# Returns
- `String`: Grouped fractional-part string.

# Errors
- No explicit validation is performed.

# Notes
- This helper is purely visual and does not change the numerical meaning.
"""
@inline function _group_digits_left(
    s::AbstractString;
    group::Int = 3,
    sep::AbstractString = raw"\,"
)::String
    n = length(s)
    n ≤ group && return String(s)

    io = IOBuffer()
    for i in 1:n
        write(io, s[i])
        if (i % group == 0) && (i < n)
            write(io, sep)
        end
    end
    return String(take!(io))
end

"""
    latex_group_fraction_digits(
        avgerr::AbstractString;
        group::Int = 3,
        sep::AbstractString = raw"\\,"
    ) -> String

Apply [``\\LaTeX``](https://www.latex-project.org/)-style digit grouping to the central-value portion of an uncertainty string.

# Function description
This helper takes a formatted uncertainty string such as `"2.00000000000000(39)"`
or `"123456.789012(34)"` and inserts digit-group separators into the central
value for improved visual readability.

Grouping is applied separately to:

- the integer part, from right to left,
- the fractional part, from left to right.

The uncertainty digits inside parentheses are left unchanged.

# Arguments
- `avgerr::AbstractString`: Input uncertainty string.
- `group::Int`: Number of digits per visual group.
- `sep::AbstractString`: Separator inserted between groups.

# Returns
- `String`: Reformatted string with grouped digits in the central value.

# Errors
- No explicit validation is performed.
- If the input does not contain `"("`, the original string is returned unchanged.

# Notes
- This helper is intended for presentation only.
- A trailing `" *"` marker is preserved if present.
"""
function latex_group_fraction_digits(
    avgerr::AbstractString;
    group::Int = 3,
    sep::AbstractString = raw"\,"
)::String
    s = String(avgerr)

    # Preserve trailing " *" if present (and any surrounding spaces)
    star = ""
    if endswith(s, " *")
        s = s[1:end-2]
        star = " *"
    end

    # Split central vs "(...)" part
    lp = findfirst(==('('), s)
    lp === nothing && return String(avgerr)  # unexpected format; fail safe

    central = s[1:prevind(s, lp)]
    tail = s[lp:end]

    # Handle optional scientific notation in the central part
    epos = findfirst(c -> (c == 'e' || c == 'E'), central)
    mant = epos === nothing ? central : central[1:prevind(central, epos)]
    expo = epos === nothing ? ""      : central[epos:end]

    # Only group digits in the fractional part of mantissa
    dot = findfirst(==('.'), mant)
    dot === nothing && return mant * expo * tail * star

    intpart = mant[1:prevind(mant, dot)]
    fracpart = mant[nextind(mant, dot):end]

    intpart_grouped = _group_digits_right(intpart; group=group, sep=sep)
    fracpart_grouped = _group_digits_left(fracpart; group=group, sep=sep)

    if length(intpart) ≤ group && length(fracpart) ≤ group
        return intpart * "." * fracpart * expo * tail * star
    end

    io = IOBuffer()
    write(io, intpart)
    write(io, ".")

    n = lastindex(fracpart)
    i = firstindex(fracpart)
    k = 0
    while i ≤ n
        write(io, fracpart[i])
        k += 1
        if (k % group == 0) && (i != n)
            write(io, sep)
        end
        i = nextind(fracpart, i)
    end

    return intpart_grouped * "." * fracpart_grouped * expo * tail * star
end

"""
    avgerr_e2d_from_float(
        cen::Float64,
        err::Float64;
        latex_grouping::Bool = false
    ) -> String

Format a central value and uncertainty from `Float64` inputs.

# Function description
This is a convenience wrapper around [`avgerr_e2d`](@ref). It first converts
both inputs to `%.14e` scientific-notation strings, then passes them to the
main formatter.

Optionally, the result can be post-processed by
[`latex_group_fraction_digits`](@ref) for [``\\LaTeX``](https://www.latex-project.org/)-style digit grouping.

# Arguments
- `cen::Float64`: Central value.
- `err::Float64`: Error value.

# Keyword arguments
- `latex_grouping::Bool`: Whether to apply digit grouping to the central-value
  portion of the formatted result.

# Returns
- `String`: Formatted uncertainty string.

# Errors
- Inherits the failure-tolerant behavior of [`avgerr_e2d`](@ref).

# Notes
- Grouping affects only the visual representation of the central value.
- The uncertainty digits inside parentheses are left unchanged.
"""
function avgerr_e2d_from_float(
    cen::Float64, 
    err::Float64;
    latex_grouping::Bool = false
)::String
    cen_str = @sprintf("%.14e", cen)
    err_str = @sprintf("%.14e", err)

    base = avgerr_e2d(cen_str, err_str)
    return latex_grouping ? latex_group_fraction_digits(base) : base    
end

end  # module AvgErrFormatter