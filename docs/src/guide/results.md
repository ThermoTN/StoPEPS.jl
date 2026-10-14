# Reading results

After the [2 × 2 TFIM calculation](https://github.com/ThermoTN/StoPEPS.jl#a-quick-start),
run this in Julia from the repository root with `--project=.` to check the
result and print its curve:

```julia
using StoPEPS
include("examples/transverse_Ising_model/config.quickstart.jl")
c = quickstart_config()
r = load_result(joinpath(c.output_directory, "results", "result.jld2"))
@assert r["metadata"]["complete"]
rows = r["observables"]["curve"]
@assert [row.beta for row in rows] == temperature_nodes(c)
@assert all(row -> isfinite(row.energy_per_site) && isfinite(row.energy_per_site_error), rows)
for row in rows
    println((beta=row.beta, energy=row.energy_per_site,
             stderr=row.energy_per_site_error, lnZ=row.lnZ, eta=row.eta))
end
```

The curve contains purification points below beta0 and stochastic estimates
from beta0 onward. A resampling point contains the estimate after the event.
Energy is measured at `beta_measure` nodes; other nodes store `NaN` for energy.

## Completion and progress

For preparation, `metadata["stage"]=="purification"` and
`metadata["stage_complete"]==true` establish stage success. `complete` becomes
true when stochastic evolution finishes. For an interrupted calculation,
inspect `metadata["progress"]`, `get(metadata,"error",nothing)` and the last
collected curve row. [Execution and output](../design/runtime.md) lists the
files and beta0 restart procedure.

## Numerical fields

The top-level object and its `parameters`, `metadata`, `observables` and
`diagnostics` are string-keyed dictionaries. Curve and sample entries are
named tuples accessed with dot syntax.

| Field | Meaning |
|---|---|
| `r["schema_version"]` | Numerical format version; see the [result format](https://github.com/ThermoTN/StoPEPS.jl/blob/main/docs/result-schema.md) |
| `r["parameters"]` | Effective configuration, basis and contraction conventions |
| `r["metadata"]` | Stage, completion, progress, run identity and execution settings |
| `r["observables"]["purification"]` | Thermal preparation rows through beta0 |
| `r["observables"]["curve"]` | Main thermal curve |
| `row.beta` | Inverse temperature |
| `row.energy_per_site`, `row.energy_per_site_error` | Energy per site and sampling standard error |
| `row.energy_total` | Total energy |
| `row.lnZ`, `row.lnZ_error` | Log partition function and its conditional error |
| `row.eta`, `row.eta_error` | Sampling efficiency and its conditional error |
| `row.max_state_truncation` | Maximum full-update relative squared fit residual |
| `row.max_environment_truncation` | Boundary contraction diagnostic, defined below |
| `row.max_boundary_entropy` | Maximum boundary entropy during the evolution interval, in nats |
| `row.time_seconds` | Evolution wall time for the interval |
| `r["observables"]["samples"]` | Sample configurations, multiplicities, weights, energies and diagnostics at stochastic output nodes |
| `r["observables"]["replay"]` | Regenerated-sample evolution records |
| `r["diagnostics"]["events"]` | Resampling events, selected mixture proportions and sample records |
| `r["diagnostics"]["history"]` | Tensor normalization, evolution and solver diagnostics |

See [physical conventions](../design/physical_conventions.md) for the Hamiltonian.
`nothing` denotes an optional field or inapplicable measurement. `NaN` denotes
an unmeasured observable or undefined diagnostic, including initial efficiency
and entropy values. Preserve these values when analyzing output.

## Sample records

Each sample record carries `beta`, `id`, `seed`, `sample_index`, `configuration`,
`count`, `log_weight`, `log_p0`, `log_q_proposal`, `physical_log_norm`,
`energy_per_site` and contraction diagnostics.

The main curve provides the weighted estimates and their sampling errors.
See [thermal weights](../design/finite_temperature.md) for the normalization
and [convergence](validation.md) for uncertainty analysis.

## Contraction diagnostics and entropy

| Method | `max_environment_truncation` | Sample/preparation `Se` layout |
|---|---|---|
| BoundaryMPSRow | Canonical SVD relative discarded squared norm | `north[y,b]`, `south[y,b]`, shape (Ly-1,Lx-1) |
| BoundaryMPSDiagonal | Canonical SVD relative discarded squared norm | `northwest[x,y]`, `southeast[x,y]`, shape (Lx,Ly) |
| BoundaryMPSRowVariational | Local projected SVD relative discarded squared norm | `north[y,b]`, `south[y,b]`, shape (Ly-1,Lx-1) |
| BoundaryMPSDiagonalVariational | Relative squared whole-layer fit residual | `northwest_cuts`, `southeast_cuts`, records of layer, left_edge, right_edge and entropy |

Entropy profiles are recorded during energy measurement when
`record_boundary_entropy=true`. They describe the contraction boundary MPS.
Row indices identify the absorbed row and cut after column b; diagonal indices
identify the two-output split at a lattice vertex. Variational profiles use
the projected two-site spectra from the accepted fit. Unmeasured profiles
are `nothing`; cuts with undefined spectra store `NaN`.

For the row-contraction quick start, for example:

```julia
sample = first(r["observables"]["samples"])
println(sample.Se.north)
```

`E.E.` in the logs is `max_boundary_entropy` from the evolution interval.
The main log also reports full-update iteration and linear-solver summaries.
Use [troubleshooting](troubleshooting.md) to interpret them.
