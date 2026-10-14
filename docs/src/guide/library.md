# Library tutorials

Run these snippets from the repository root in Julia with `--project=.`.
The [API reference](../api.md) describes the public entry points.

## A complete serial calculation

```julia
using StoPEPS, LinearAlgebra
BLAS.set_num_threads(1)
include("examples/transverse_Ising_model/config.quickstart.jl")
c = quickstart_config(output_directory=nothing)
checkpoint = run_purification(c)
result = run_stochastic(c; checkpoint)
@assert result["metadata"]["complete"]
for row in result["observables"]["curve"]
    println((row.beta, row.energy_per_site, row.energy_per_site_error))
end
```

`output_directory=nothing` keeps this calculation in memory. A path enables
the logs, checkpoint and numerical result described in [runtime](../design/runtime.md).

## Draw a Born sample

Using the checkpoint above, construct a sampling environment and an explicit
random stream:

```julia
boundary = BoundaryMPSRow(c.chi_sampling)
cache = SamplingCache(checkpoint.state, boundary; basis=c.sampling_basis)
draw = sample_configuration(cache; rng=sample_rng(c.seed, 1))
sample, normalizations = project_configuration(
    checkpoint.state, draw.configuration; basis=c.sampling_basis,
)
measurement = measure_energy(sample, c.model, BoundaryMPSRow(c.chi_measurement))
println((configuration=draw.configuration, log_probability=draw.log_probability,
         energy_per_site=measurement.energy_per_site))
```

This prints one Born sample's expectation value. The thermal curve combines
samples with the weights described in [thermal evolution](../design/finite_temperature.md).
`sample_configuration(cache;configuration=draw.configuration)` evaluates the
sampling probability of that configuration.

## Compare exact thermodynamics

```julia
for row in result["observables"]["curve"]
    reference = exact_thermal_observables(c.lattice, c.model, row.beta)
    println((beta=row.beta, energy=row.energy_per_site,
             exact=reference.energy_per_site,
             difference=row.energy_per_site-reference.energy_per_site,
             stderr=row.energy_per_site_error))
end
```

The dense reference returns energy per site, total energy and lnZ.
Hilbert-space storage grows
exponentially with site count; the four-site quick start provides a small
reference for the workflow.

## Reuse a saved checkpoint

```julia
using StoPEPS
include("examples/transverse_Ising_model/config.quickstart.jl")
c = quickstart_config(output_directory="output/independent-run", seed=42)
path = "examples/transverse_Ising_model/output/quickstart/results/checkpoint.jld2"
result = run_stochastic(c; checkpoint_path=path)
```

For MPI, initialize MPI and call `run_stochastic(c;checkpoint_path=path,
comm=MPI.COMM_WORLD)` on every rank. Rank zero returns the result dictionary.
The [launch guide](quickstart.md) gives the application commands.
