# Maranatha.Utils.MaranathaIO

## Overview

`Maranatha.Utils.MaranathaIO` provides result-serialization, summary-export,
and datapoint-level file-management helpers for `Maranatha.jl`.

Its responsibilities are broader than simple file saving. It also supports:

- structured round-trip conversion between internal result objects and
  serialization-friendly dictionaries,
- human-readable [`TOML`](https://toml.io/en/) summaries,
- subdivision-count handling, including direct use of stored `nsamples` and
  reconstruction from step-size information when needed,
- merging partial result blocks while preserving aligned datapoint arrays,
- dropping selected datapoints from existing result files.


---

## Serialization layer

The round-trip pair

- [`Maranatha.Utils.MaranathaIO.namedtuple_to_dict`](@ref)
- [`Maranatha.Utils.MaranathaIO.dict_to_namedtuple`](@ref)

forms the structured serialization boundary of this module.

The first converts internal result objects into dictionaries that are easier to
store safely. The second reconstructs the expected internal result layout after
loading, restoring `nsamples` directly when available and inferring them from
step-size data when needed.

This keeps the external storage representation separate from the in-memory
analysis representation while preserving the result shape expected by later
analysis utilities.

---

## Summary export

The helper [`Maranatha.Utils.MaranathaIO.generate_summary_dict`](@ref) builds a human-readable summary view
intended for [`TOML`](https://toml.io/en/) output.

This summary is not primarily meant for exact round-trip reconstruction. Its
purpose is quick inspection of:

- metadata,
- stored or reconstructed subdivision counts,
- sampled step sizes, including `tuple_h` when available,
- averages,
- error totals,
- detailed error decomposition.

That makes it a useful companion to binary `JLD2` output.

---

## Save / load entry points

The primary storage helpers are:

- [`Maranatha.Utils.MaranathaIO.save_datapoint_results`](@ref)
- [`Maranatha.Utils.MaranathaIO.load_datapoint_results`](@ref)

These define the standard on-disk contract for Maranatha datapoint results.

The current design assumes `.jld2` as the authoritative storage format, with an
optional `.toml` summary beside it.

---

## Inferred sample-count utilities

Because result objects may already store `nsamples` or may require those counts
to be reconstructed from `h` / `tuple_h`, this module includes helpers such as:

- [`Maranatha.Utils.MaranathaIO.infer_nsamples`](@ref)
- [`Maranatha.Utils.MaranathaIO._build_nsamples_suffix`](@ref)
- [`Maranatha.Utils.MaranathaIO._default_result_path`](@ref)

These utilities support direct reuse of stored counts, reconstruction when
needed, filename construction, and later dataset editing workflows.

---

## Merge workflow

The merge helpers

- [`Maranatha.Utils.MaranathaIO._assert_same_result_shape`](@ref)
- [`Maranatha.Utils.MaranathaIO._assert_no_duplicate_h`](@ref)
- [`Maranatha.Utils.MaranathaIO.merge_datapoint_results`](@ref)
- [`Maranatha.Utils.MaranathaIO.merge_datapoint_result_files`](@ref)

support the common use case where a larger convergence run was produced in
several partial batches and later needs to be combined into one dataset.

The merge policy is intentionally conservative:

- global metadata must match,
- datapoint arrays, including `nsamples`, must remain aligned,
- duplicate `h` values are rejected by default.

This keeps merging predictable and reduces ambiguity while preserving a merged
result object with explicit subdivision counts.

---

## Datapoint-removal workflow

The filtering helpers

- [`Maranatha.Utils.MaranathaIO.drop_nsamples_from_result`](@ref)
- [`Maranatha.Utils.MaranathaIO.drop_nsamples_from_file`](@ref)

allow selected subdivision counts to be removed from an already-generated
dataset.

These helpers use stored `nsamples` when available and otherwise rebuild them
from step-size information before filtering.

This is particularly useful when preparing custom fitting subsets or when a
small number of datapoints should be excluded without recomputing the entire
dataset.

---

## Design scope

`Maranatha.Utils.MaranathaIO` is about **result persistence and dataset
management**.

It does **not**:

- perform quadrature,
- estimate derivatives,
- fit extrapolation models,
- validate the mathematical correctness of the underlying result.

Instead, it handles the infrastructure around storing, reconstructing, merging,
and filtering already-computed result objects.

---

## API reference

```@autodocs
Modules = [
    Main.Maranatha.Utils.MaranathaIO,
]
Private = true
```