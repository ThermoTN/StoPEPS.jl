# Result format

`StoPEPS.Results` reads and writes versioned JLD2 results. The current
schema is `2.7.0`; `RESULT_SCHEMA_VERSION` exposes it in Julia.
The [reading guide](src/guide/results.md) gives executable examples and
field definitions.

The top-level keys `schema_version`, `created_at`, `method` and `model` record
the format version, creation time and calculation identity. The reading guide
defines the `metadata`, `parameters`, `observables` and `diagnostics` groups.

`results/checkpoint.jld2` stores the thermal PEPO at `beta0` in
`observables["sites"]`, with `beta0` and `log_amplitude`.
`results/result.jld2` stores numerical measurements and diagnostics.
Each write atomically publishes the file.

FU algorithm revision 2 adds optional `full_update_algorithm*` parameter fields
and scalar `fu_gauge` summaries in evolution histories. These identify the
environment and internal bond gauges without storing tensors or changing
schema `2.7.0`. See [gauge diagnostics](src/guide/results.md#full-update-gauge-diagnostics).

See [runtime](src/design/runtime.md) for output timing and the reading guide
for completion status and sample records.
