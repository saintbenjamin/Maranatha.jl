"""
    _fmt_sci_texttt(x::Real) -> String

Format a real number in fixed scientific notation.

# Function description

This helper converts `x` to a floating-point value and formats it using
six-digit scientific notation:

    "%.6e"

It is primarily used as a low-level building block for text-based reporting,
where a consistent machine-readable numeric format is required.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: Scientific-notation representation of `x`.

# Errors

- No explicit validation is performed.
- Non-finite values (`NaN`, `Inf`) are passed through to `Printf`.

# Notes

- The output does **not** include any [``\\LaTeX``](https://www.latex-project.org/) or Markdown markup.
- Intended for internal use by higher-level formatting helpers.
"""
function _fmt_sci_texttt(x::Real)
    return @sprintf("%.6e", float(x))
end

"""
    _fmt_tex_texttt_sci(x::Real) -> String

Format a number in [``\\LaTeX``](https://www.latex-project.org/) monospace scientific notation.

# Function description

This helper wraps the scientific-notation output of
[`_fmt_sci_texttt`](@ref) inside a [``\\LaTeX``](https://www.latex-project.org/) `\\texttt{}` command:

    \\texttt{1.234567e-03}

It is intended for inclusion in [``\\LaTeX``](https://www.latex-project.org/) tables or inline text where a
monospaced numeric appearance improves readability.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: [``\\LaTeX``](https://www.latex-project.org/)-formatted scientific-notation string.

# Errors

- No explicit validation is performed.

# Notes

- The returned string is safe for use in [``\\LaTeX``](https://www.latex-project.org/) document bodies.
- No math-mode delimiters (`\$`) are added automatically.
"""
function _fmt_tex_texttt_sci(x::Real)
    return "\\texttt{$(_fmt_sci_texttt(x))}"
end

"""
    _fmt_md_code_sci(x::Real) -> String

Format a number as Markdown inline code in scientific notation.

# Function description

This helper wraps the scientific-notation output of
[`_fmt_sci_texttt`](@ref) inside Markdown backticks:

    `1.234567e-03`

It is suitable for README files, reports, or logs rendered using
Markdown engines.

# Arguments

- `x::Real`: Numeric value to format.

# Returns

- `String`: Markdown inline-code representation.

# Errors

- No explicit validation is performed.

# Notes

- Intended for human-readable documentation rather than machine parsing.
"""
function _fmt_md_code_sci(x::Real)
    return "`$(_fmt_sci_texttt(x))`"
end

"""
    _fmt_avgerr_tex(x::Real, err::Real) -> String

Format a central value and uncertainty for [``\\LaTeX``](https://www.latex-project.org/) output.

# Function description

This helper attempts to produce a compact parenthetical uncertainty string
using

    AvgErrFormatter.avgerr_e2d_from_float(...; latex_grouping=true)

Examples:

- `1.23(45)`
- `2.000\\,000\\,000(39)`

If the compact formatting fails for any reason, the function falls back to a
simple `x ± err` representation using [``\\LaTeX``](https://www.latex-project.org/)-compatible syntax:

    "%.7g \\pm %.2g"

# Arguments

- `x::Real`: Central value.
- `err::Real`: Uncertainty.

# Returns

- `String`: Formatted uncertainty string suitable for [``\\LaTeX``](https://www.latex-project.org/).

# Errors

- Formatting failures are caught internally; no exception is propagated.

# Notes

- Digit grouping may be applied to long mantissas.
- The fallback representation prioritizes robustness over compactness.
"""
function _fmt_avgerr_tex(x::Real, err::Real)
    return try
        AvgErrFormatter.avgerr_e2d_from_float(float(x), float(err); latex_grouping=true)
    catch
        @sprintf("%.7g \\pm %.2g", float(x), float(err))
    end
end

"""
    _fmt_avgerr_md(x::Real, err::Real) -> String

Format a central value and uncertainty for Markdown output.

# Function description

This helper is analogous to [`_fmt_avgerr_tex`](@ref) but produces
Markdown-friendly output.

It first attempts compact parenthetical formatting via

    AvgErrFormatter.avgerr_e2d_from_float(...; latex_grouping=false)

If that fails, it falls back to a Unicode plus–minus representation:

    "%.7g ± %.2g"

# Arguments

- `x::Real`: Central value.
- `err::Real`: Uncertainty.

# Returns

- `String`: Formatted uncertainty string suitable for Markdown text.

# Errors

- No exception is propagated from formatting failures.

# Notes

- Digit grouping is disabled to avoid introducing [``\\LaTeX``](https://www.latex-project.org/)-specific markup.
- Intended for reports, README files, or console-rendered Markdown.
"""
function _fmt_avgerr_md(x::Real, err::Real)
    return try
        AvgErrFormatter.avgerr_e2d_from_float(float(x), float(err); latex_grouping=false)
    catch
        @sprintf("%.7g ± %.2g", float(x), float(err))
    end
end

"""
    _latex_escape_underscore(s::AbstractString) -> String

Escape underscores for safe use in [``\\LaTeX``](https://www.latex-project.org/) text mode.

# Function description

This helper replaces every underscore character `_` with the [``\\LaTeX``](https://www.latex-project.org/)-escaped
sequence `\\_`, preventing compilation errors in text contexts such as
captions, labels, or section titles.

# Arguments

- `s::AbstractString`: Input string.

# Returns

- `String`: String with underscores escaped for [``\\LaTeX``](https://www.latex-project.org/).

# Errors

- No explicit validation is performed.

# Notes

- Only underscores are handled; other special characters are not escaped.
- Intended for simple identifiers (file names, labels, rule names).
"""
function _latex_escape_underscore(s::AbstractString)
    return replace(s, "_" => "\\_")
end