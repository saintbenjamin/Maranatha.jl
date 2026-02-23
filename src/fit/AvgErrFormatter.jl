# ============================================================================
# src/fit/AvgErrFormatter.jl (Benji: taken from src/Sarah/AvgErrFormatter.jl of Deborah.jl)
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module AvgErrFormatter

import Printf: @sprintf

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
    avgerr_e2d(
        censtr::String, 
        errstr::String
    ) -> String

Format a central value and its error into a compact string using parenthetical ``\\pm`` notation.

This function takes the central value and error as strings in scientific notation (e.g., `"1.234e+01"`, `"3.2e-01"`), 
and returns a formatted string such as `"12.3(3)"`. It handles different scales and adds a `*` suffix if the error dominates.

# Arguments
- `censtr::String` : Central value string in scientific notation.
- `errstr::String` : Error value string in scientific notation.

# Returns
- `String` : Formatted result like `"1.23(45)"` or `"1.2(3) *"` depending on magnitude and dominance.
"""
function avgerr_e2d(
    censtr::String, 
    errstr::String
)::String
    cen = parse(Float64, censtr)
    err = parse(Float64, errstr)

    # Special case: error much bigger than central value
    if err ≥ 1.0 && abs(cen) < 0.1 * err
        expo = floor(Int, log10(abs(err)))
        cen_mantissa = cen / 10.0^expo
        err_scaled = round_sig(err / 10.0^expo, 2)
        err_2digit = err_scaled < 10 ? round(Int, err_scaled * 10) : round(Int, err_scaled)

        return @sprintf("%.1fe+%02d(%02d) *", cen_mantissa, expo, err_2digit)
    end

    if err < 1.0
        digit0 = parse(Int, split(errstr, "-")[2]) - 1
        digit = digit0 + 2

        cen_round_str = @sprintf("%.*f", digit, cen)
        err_round = round_sig(err)
        err_round_int = Int(round(err_round * 10^digit))

        if abs(cen) > err
            return @sprintf("%s(%d)", cen_round_str, err_round_int)
        else
            return @sprintf("%s(%d) *", cen_round_str, err_round_int)
        end

    elseif 1.0 ≤ err < 10.0
        if abs(cen) > 1.0
            digit = 1
            cen_round_str = @sprintf("%.*f", digit, cen)
            err_round = round_sig(err)
            err_round_int = Int(round(err_round * 10^digit))

            if abs(cen) > err
                return @sprintf("%s(%d)", cen_round_str, err_round_int)
            else
                return @sprintf("%s(%d) *", cen_round_str, err_round_int)
            end
        else
            digit0 = parse(Int, split(censtr, "-")[2])
            digit = digit0 + 2

            cen_round_str = @sprintf("%.*f", digit0, cen)
            err_round = round_sig(err, digit)
            err_round_int = Int(round(err_round * 10^digit0))

            if abs(cen) > err
                return @sprintf("%s(%d)", cen_round_str, err_round_int)
            else
                return @sprintf("%s(%d) *", cen_round_str, err_round_int)
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
                return @sprintf("%d(%d)", cen_round, err_round_int)
            else
                return @sprintf("%d(%d) *", cen_round, err_round_int)
            end
        else
            digit0 = parse(Int, split(censtr, "-")[2])
            digit = digit0 + 2

            cen_round_str = @sprintf("%.*f", digit0, cen)
            err_round = round_sig(err, digit)
            err_round_int = Int(round(err_round * 10^digit0))

            if abs(cen) > err
                return @sprintf("%s(%d)", cen_round_str, err_round_int)
            else
                return @sprintf("%s(%d) *", cen_round_str, err_round_int)
            end
        end
    end
end

"""
    avgerr_e2d_from_float(
        cen::Float64, 
        err::Float64
    ) -> String

Convert a central value and its error from `Float64` into a compact exponential ``\\pm`` string.

This is a wrapper for [`avgerr_e2d`](@ref) that takes float inputs and internally converts them to
scientific notation strings with high precision before formatting.

# Arguments
- `cen::Float64` : Central value.
- `err::Float64` : Error value.

# Returns
- `String` : Formatted output like `"1.23(4)e+02"` or `"1.2(3) *"`.
"""
function avgerr_e2d_from_float(
    cen::Float64, 
    err::Float64
)::String
    cen_str = @sprintf("%.14e", cen)
    err_str = @sprintf("%.14e", err)
    return avgerr_e2d(cen_str, err_str)
end

end  # module AvgErrFormatter