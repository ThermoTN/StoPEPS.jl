# Execution and output

The applications share one driver. Library calls run serially by default;
MPI execution loads `StoPEPSMPIExt` when MPI is initialized and a communicator
is passed to `run_stochastic`. Rank zero owns the main log and numerical
result. Each rank evolves its assigned samples and writes their logs.

## Stages

| Stage | Library call | Output |
|---|---|---|
| purification | `run_purification(config)` | ThermalCheckpoint at beta0, preparation curve and logs |
| stochastic | `run_stochastic(config; checkpoint, checkpoint_path, comm)` | Born samples, thermal curve, sample records and resampling events |

Preparation runs in a single process. Stochastic execution accepts an
in-memory checkpoint or a checkpoint path; its default path is
`output_directory/results/checkpoint.jld2`. MPI broadcasts the checkpoint
and distributes the samples. Rank zero returns the numerical result dictionary.

Checkpoint reuse checks the lattice, model, physical representation and beta0.
Each stochastic run can choose its sample count, seed, basis, continuation
dimensions and measurement grid. Its result records the preparation parameters.
An interrupted calculation restarts from beta0.

## Files

Paths are relative to `output_directory`. Setting it to `nothing` keeps a
library calculation in memory.

| Path | Contents |
|---|---|
| results/checkpoint.jld2 | beta0 PEPO tensors, spaces, normalization and preparation records |
| results/result.jld2 | Numerical curves, sample records, diagnostics and completion metadata |
| log/run.log | Parameters and preparation, sampling, evolution and resampling tables |
| log/samples/rank_XXX_sample_XXX.log | Sample trajectories on each rank |

Numerical results follow the
[result format](https://github.com/ThermoTN/StoPEPS.jl/blob/main/docs/result-schema.md).
They are saved at stage completion and periodically according to
`save_interval_seconds`. A failed calculation retains its collected records
and error text. See [reading results](../guide/results.md)
for status checks.

`overwrite_output=true`, or CLI `--overwrite`, permits replacing an existing
calculation. A matching stochastic stage appends to the preparation log.
Direct checkpoint writes use `save_checkpoint(path,checkpoint;overwrite=true)`
for replacement.

Use the [architecture](https://github.com/ThermoTN/StoPEPS.jl/blob/main/docs/architecture.md)
for the library structure and the [library tutorials](../guide/library.md)
and [API reference](../api.md) for public entry points.
