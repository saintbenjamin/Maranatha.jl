# ============================================================================
# src/integrands/F0000.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module F0000Preset

using ..Integrands
using ..F0000GammaEminus1

export __register_F0000_integrand__

# ============================================================
# Callable wrapper
# ============================================================

"""
  struct F0000Integrand

Callable wrapper integrand for the F0000 computation on `t ∈ [0, 1]`.

# struct description
This struct provides a user-friendly, callable integrand object that can be
passed directly to `Maranatha.Runner.run_Maranatha`. Internally, it delegates to

`F0000GammaEminus1.gtilde_F0000(t; p=p, eps=eps)`,

which implements the transformed integrand after the substitution `y = t^p`
for the original F0000 integral.

# Fields
- `p::Int`: Power in the substitution `y = t^p` (integer, typically 2–4).
- `eps::Float64`: Endpoint cutoff (stored as `Float64`) used by `gtilde_F0000`
  to suppress singular behavior near `t = 0` and `t = 1`.

# Notes
- To remain compatible with automatic differentiation (e.g., `ForwardDiff.Dual`
  values of `t`), this wrapper stores `eps` as `Float64` but converts it to
  `typeof(t)` at call time.
- This preserves the original numerical behavior for normal `Float64` usage
  while preventing keyword-type mismatches under AD.
"""
struct F0000Integrand
    p::Int
    eps::Float64
end

"""
    (f::F0000Integrand)(t)

Evaluate the F0000 transformed integrand at `t`.

# Arguments
- `t`: Real scalar in `[0, 1]` (may also be a dual number under AD).

# Returns
- The value of `g̃(t)` as returned by
  `F0000GammaEminus1.gtilde_F0000(t; p=f.p, eps=convert(typeof(t), f.eps))`.

# Notes
- `eps` is converted to `typeof(t)` to keep the keyword argument type-consistent
  when `t` is a `ForwardDiff.Dual`.
"""
function (f::F0000Integrand)(t)
    eps_t = convert(typeof(t), f.eps)
    return F0000GammaEminus1.gtilde_F0000(t; p=f.p, eps=eps_t)
end

# ============================================================
# Factory
# ============================================================

"""
    factory_F0000(; 
      p::Int=2, 
      eps::Float64=1e-15
    )

Factory for constructing the registered F0000 integrand.

# Function description
This factory is registered under the name `:F0000` in the integrand registry.
It returns a callable `F0000Integrand` instance that can be used as a standard
integrand function.

# Keyword arguments
- `p::Int=2`: Substitution power used in `y = t^p`.
- `eps::Float64=1e-15`: Endpoint cutoff forwarded to `gtilde_F0000`.

# Returns
- `F0000Integrand`: Callable integrand object.

# Notes
- `eps` is stored as `Float64` but promoted to `typeof(t)` when evaluating the
  integrand. This makes the preset safe under `ForwardDiff` while keeping the
  default behavior deterministic for typical `Float64` runs.
"""
function factory_F0000(; 
  p::Int=2, 
  eps::Float64=1e-15
)
    return F0000Integrand(p, eps)
end

# ============================================================
# Registration hook
# ============================================================

"""
    __register_F0000_integrand__()

Register the `:F0000` integrand factory into `Maranatha.Integrands`.

# Function description
This function installs the factory `factory_F0000` into the project-wide
integrand registry, enabling the user-facing construction:

`Maranatha.Integrands.integrand(:F0000; p=..., eps=...)`.

# Returns
- `nothing`

# Notes
- This is intended to be called once during package/module initialization
  (e.g., from `src/Maranatha.jl` after including this file).
"""
function __register_F0000_integrand__()
    Integrands.register_integrand!(:F0000, factory_F0000)
    return nothing
end

end # module F0000Preset