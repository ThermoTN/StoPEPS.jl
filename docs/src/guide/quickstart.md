# Getting started

Use Julia 1.11 or later and run the commands from the repository root.

## Serial quick start

The [README](https://github.com/ThermoTN/StoPEPS.jl#a-quick-start)
contains the complete 2 × 2 open-boundary TFIM calculation. The root environment
provides StoPEPS and its numerical dependencies; the example uses these settings:

| Setting | Quick-start value |
|---|---|
| Lattice and model | 2 × 2 square lattice with open boundaries; TFIM, J=1 and h=0.7, Pauli operators |
| Sampling | X basis, Ns=32, seed=20260924 |
| Temperature grid | beta=[0, 0.1, 0.2, 0.3], beta0=0.1 |
| Bond dimensions | PEPO/PEPS D=4; boundary-MPS chi=16 |
| Evolution | Full update, second-order Trotter steps, delta_tau=0.025 |
| Contraction | Row boundary MPS (`BoundaryMPSRow`) |
| Ensemble evolution | Direct evolution of the initial Born samples |

Results are written to `examples/transverse_Ising_model/output/quickstart/`.
`results/checkpoint.jld2` stores the `beta0` PEPO, and `results/result.jld2`
stores the numerical curve. A successful run has four curve rows and
`metadata["complete"]==true`; [reading results](results.md) shows how to check
these and print energy per site with its sampling error.

Julia startup and compilation contribute to whole-process wall time;
`log/run.log` reports evolution-step times separately. Measure a larger system
before choosing cluster resources.

`quickstart_config(output_directory="output/run2", seed=42)` selects a fresh
directory and seed. Set `overwrite_output=true` to replace an existing run.

## Application configuration

Each application has a `config.jl` defining `config::RunConfig` and
`blas_threads`. Copy it to `config.local.jl` for local calculations:

```sh
APP=examples/transverse_Ising_model
cp -n "$APP/config.jl" "$APP/config.local.jl"
```

Edit `config.local.jl` to set the model, numerical controls and
`output_directory`. The entry point loads that complete file when present,
otherwise it loads `config.jl`. Git ignores local configurations and output.
Both stages and all MPI ranks use the selected configuration.

The [TFIM example README](https://github.com/ThermoTN/StoPEPS.jl/blob/main/examples/transverse_Ising_model/README.md#parameters)
lists the Hamiltonian, supplied parameters and temperature grids.
Its `config.jl` writes to the example directory by default;
set `output_directory` in the local configuration to keep separate runs.

For a small MPI calculation, the TFIM `config.local.jl` can contain:

```julia
include(joinpath(@__DIR__, "config.quickstart.jl"))
const blas_threads = 1
const config = quickstart_config(
    output_directory=joinpath(@__DIR__, "output", "mpi-quickstart"),
)
```

## Application environments and MPI

The application environment adds MPI and loads this checkout through `[sources]`.
Instantiate it and install the matching launcher:

```sh
APP=examples/transverse_Ising_model
julia --project="$APP" -e 'using Pkg; Pkg.instantiate()'
julia --project="$APP" -e 'using MPI; MPI.install_mpiexecjl()'
```

Add the depot's `bin` directory to PATH or use the installed `mpiexecjl` path.
Then run the preparation and stochastic commands in the
[TFIM example](https://github.com/ThermoTN/StoPEPS.jl/blob/main/examples/transverse_Ising_model/README.md#run).

Use `-n 1` for a single-rank calculation. The stochastic entry point initializes
MPI; each rank receives the beta0 checkpoint. Rank zero writes the main result.
Choose a shared output directory for multi-node work. A system MPI installation
uses a matching launcher and Julia MPI backend; see [troubleshooting](troubleshooting.md).

## Cluster jobs

The two stages can run as dependent Slurm jobs. Adjust resources and Julia/MPI
paths to the cluster, and submit from the shared repository directory:

```sh
prep=$(sbatch --parsable --job-name=stopeps-prep --ntasks=1 --cpus-per-task=1 \
  --wrap='julia --project=examples/transverse_Ising_model examples/transverse_Ising_model/main.jl --stage=purification')
sbatch --dependency=afterok:$prep --job-name=stopeps-samples --ntasks=4 --cpus-per-task=1 \
  --wrap='mpiexecjl --project=examples/transverse_Ising_model -n 4 julia --project=examples/transverse_Ising_model examples/transverse_Ising_model/main.jl --stage=stochastic'
```

Preparation writes the beta0 checkpoint; sampling starts after that job
succeeds. See [parameters](parameters.md) and [convergence](validation.md) for
choosing numerical settings.
