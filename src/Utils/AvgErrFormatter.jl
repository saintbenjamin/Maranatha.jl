# ============================================================================
# src/AvgErrFormatter/AvgErrFormatter.jl
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
        sig::Int=2
    ) -> Float64

Round a number to a specified number of significant digits.

# Arguments
- `x::Float64`  : Number to round.
- `sig::Int=2`  : Number of significant digits (default: `2`).

# Returns
- `Float64` : The number rounded to the specified significant digits.
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

Construct a last-resort fallback representation of a central value and its error.

This helper is used when the normal compact parenthetical formatter fails,
for example due to unexpected exponent parsing, formatting mismatches, or
other runtime issues inside [`avgerr_e2d`](@ref).

Instead of throwing an exception, it returns both the central value and the
error explicitly in scientific notation:

```julia
"%.14e(%.14e)"
```

This is not the usual compact uncertainty notation like `"1.23(45)"`,
but it guarantees that the numerical content is still preserved in a
readable and loss-resistant form.

# Arguments

* `cen::Real` : Central value.
* `err::Real` : Error value.

# Returns

* `String` : Fallback string of the form `"1.23456789012345e+00(6.78901234567890e-03)"`.

# Notes

This routine is intended purely as a safety fallback and should only be used
when the primary formatting path cannot produce a valid compact result.
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

This helper wraps a dynamic formatting call to `Printf.format` and prevents
formatting failures from terminating the caller. If the provided format string
or arguments are incompatible, the function silently falls back to
[`_avgerr_fallback_string`](@ref), preserving the numerical information.

It is mainly used inside [`avgerr_e2d`](@ref) to protect individual return paths
such as:

- `"%s(%d)"`
- `"%s(%d) *"`
- `"%d(%d)"`
- `"%.1fe+%02d(%02d) *"`

# Arguments
- `fmt::AbstractString` : Dynamic `Printf` format string.
- `args...` : Positional arguments passed to the formatter.
- `cen::Real` : Central value used for fallback output.
- `err::Real` : Error value used for fallback output.

# Returns
- `String` : Formatted uncertainty string if successful, otherwise a scientific-notation fallback string.

# Notes
This helper guards only the final string construction stage. Earlier failures
during parsing, exponent extraction, or rounding must still be handled by the
caller.
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

Format a central value and its uncertainty into compact parenthetical notation.

This routine takes the central value and error as strings, typically in scientific
notation such as `"1.23400000000000e+01"` and `"3.20000000000000e-01"`, and tries
to convert them into a compact uncertainty expression such as

```julia
"12.30(32)"
"2.00000000000000(39)"
"1.2(3) *"
```

depending on the relative size of the error and the magnitude of the central value.

The formatter applies several magnitude-dependent branches so that the printed
result remains short and readable across a wide range of cases:

* sub-unit errors (`err < 1`)
* order-one errors (`1 ≤ err < 10`)
* large errors (`err ≥ 10`)
* special dominant-error cases where the central value is much smaller than the error

If the uncertainty is comparable to or larger than the central value, a trailing
`" *"` marker is appended to indicate that the error dominates.

This function is also designed to be failure-tolerant. If any intermediate step
fails — for example during parsing, exponent extraction, significant-digit rounding,
or final string construction — it does **not** throw an error. Instead, it falls back
to a fully explicit scientific-notation form via [`_avgerr_fallback_string`](@ref),
so that numerical information is never lost.

# Arguments

* `censtr::String` : Central value string, usually in scientific notation.
* `errstr::String` : Error value string, usually in scientific notation.

# Returns

* `String` : A compact formatted uncertainty string such as `"1.23(45)"`,
  `"2.00000000000000(39)"`, or `"1.2(3) *"`. If compact formatting fails,
  a fallback string of the form `"1.23456789012345e+00(6.78901234567890e-03)"`
  is returned instead.

# Notes

* This routine assumes that `errstr` represents a nonnegative uncertainty.
* Non-finite values (`NaN`, `Inf`) or negative errors immediately trigger fallback output.
* The parsing logic for exponent-like information currently assumes the input strings
  are compatible with the scientific-notation format typically produced by
  `@sprintf("%.14e", x)`.
* The returned parenthetical digits represent the uncertainty aligned with the last
  shown digits of the central value, following standard compact error notation.

# Examples

```julia
avgerr_e2d("2.00000000000000e+00", "3.92317790547155e-13")
# -> "2.00000000000000(39)"

avgerr_e2d("1.23400000000000e+01", "3.20000000000000e-01")
# -> "12.30(32)"

avgerr_e2d("1.00000000000000e-03", "2.50000000000000e-03")
# -> "0.0010(25) *"
```

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

Insert digit-group separators into the integer part of a number string, grouping from right to left.

This helper is intended for visual formatting of the mantissa in strings such as

```julia
"1234567"      -> "1\\,234\\,567"
"-1234567"     -> "-1\\,234\\,567"
```

It preserves a leading minus sign, if present, and applies grouping only to the
remaining digits. The separator is inserted every `group` digits starting from
the least-significant side, which matches conventional thousands-style grouping.

# Arguments

* `s::AbstractString` : Integer-part string, optionally beginning with `"-"`.
* `group::Int=3` : Number of digits per group.
* `sep::AbstractString=raw"\\,"` : Separator inserted between digit groups.

# Returns

* `String` : Grouped integer-part string.

# Notes

This helper assumes that `s` is already an integer-like substring with no decimal
point or exponent marker. It does not validate whether the remaining characters
are all decimal digits.
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

Insert digit-group separators into the fractional part of a number string, grouping from left to right.

This helper is intended for ``\\LaTeX``-friendly visual formatting of long decimal tails,
for example

```julia
"00000000000000" -> "000\\,000\\,000\\,000\\,00"
"789012"         -> "789\\,012"
```

Unlike integer-part grouping, this function starts from the first fractional digit
immediately after the decimal point and inserts the separator every `group` digits
toward the right.

# Arguments

* `s::AbstractString` : Fractional-part string with no decimal point.
* `group::Int=3` : Number of digits per group.
* `sep::AbstractString=raw"\\,"` : Separator inserted between digit groups.

# Returns

* `String` : Grouped fractional-part string.

# Notes

This helper is purely visual. It does not change the numerical meaning of the
value and is especially useful when one wants to quickly see how many decimal
places remain before the uncertainty digits begin.
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

Apply ``\\LaTeX``-style digit grouping to the central-value portion of an [`avgerr_e2d`](@ref) string.

This function takes a formatted uncertainty string such as

```julia
"2.00000000000000(39)"
"123456.789012(34)"
"-123456.789012(34) *"
"1.23456789e+03(12)"
```

and inserts a separator such as `\\,` into the central value for improved visual
readability. Grouping is applied separately to:

* the integer part, from right to left
* the fractional part, from left to right

The uncertainty part inside parentheses, such as `"(39)"`, is left unchanged.
A trailing `" *"` marker is also preserved.

Examples:

```julia
"2.00000000000000(39)"
    -> "2.000\\,000\\,000\\,000\\,00(39)"

"123456.789012(34)"
    -> "123\\,456.789\\,012(34)"

"-123456.789012(34) *"
    -> "-123\\,456.789\\,012(34) *"
```

# Arguments

* `avgerr::AbstractString` : Input string produced by [`avgerr_e2d`](@ref) or a compatible formatter.
* `group::Int=3` : Number of digits per visual group.
* `sep::AbstractString=raw"\\,"` : Separator inserted between groups, typically a ``\\LaTeX`` thin space.

# Returns

* `String` : Reformatted string with grouped digits in the central value.

# Notes

* This helper is intended for presentation only.
* It assumes that the input has the general form `"central(error)"` with an optional trailing `" *"`.
* If the input does not contain `"("`, the function returns the original string unchanged.
* The uncertainty digits are intentionally not grouped, since they already carry their own positional meaning.
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

Format a central value and its uncertainty from `Float64` inputs into compact parenthetical notation.

This is a convenience wrapper around [`avgerr_e2d`](@ref). It first converts both
the central value and the error into scientific-notation strings using

```julia
@sprintf("%.14e", x)
```

and then passes those strings to the main formatter. This ensures that the internal
parsing logic in [`avgerr_e2d`](@ref) receives a consistent exponent-style input format.

Optionally, the returned string can be post-processed with
[`latex_group_fraction_digits`](@ref) to insert ``\\LaTeX``-friendly digit grouping
markers such as `\\,` into the central-value part. This is useful when many decimal
places are shown and one wants to visually identify how far the reliable digits extend
before the uncertainty digits begin.

Examples:

```julia
avgerr_e2d_from_float(2.0, 3.92317790547155e-13)
# -> "2.00000000000000(39)"

avgerr_e2d_from_float(2.0, 3.92317790547155e-13; latex_grouping=true)
# -> "2.000\\,000\\,000\\,000\\,00(39)"
```

# Arguments

* `cen::Float64` : Central value.
* `err::Float64` : Error value.
* `latex_grouping::Bool=false` : If `true`, apply ``\\LaTeX``-style digit grouping to the
  central-value part of the formatted output.

# Returns

* `String` : Formatted uncertainty string such as `"1.23(45)"`,
  `"2.00000000000000(39)"`, or `"1.2(3) *"`. If `latex_grouping=true`,
  the central-value digits are visually grouped for readability.

# Notes

* This wrapper inherits the failure-tolerant behavior of [`avgerr_e2d`](@ref).
* The input values are always converted using fixed `%.14e` scientific notation before formatting.
* Grouping, when enabled, affects only the visual representation of the central value;
  the uncertainty digits inside parentheses are left unchanged.
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