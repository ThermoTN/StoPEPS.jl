# U(1) × SU(2) Hubbard model

## Model and Hamiltonian

This example computes finite-temperature properties of the square-lattice
Hubbard model in the grand-canonical ensemble, with open boundaries and
$U(1) \times SU(2)$ charge and spin symmetry. The physical Hamiltonian is

```math
H=-t\sum_{\langle ij\rangle,\sigma}
  (c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.})
  +U\sum_i n_{i\uparrow}n_{i\downarrow}.
```

Imaginary-time evolution uses $K=H-\mu N$ at fixed chemical potential.
The reported energy is the expectation of $H$, and `lnZ` is
$\ln\mathrm{Tr}e^{-\beta K}$. The supplied choice $\mu=U/2$ gives half
filling on average on this bipartite lattice. Born samples span different
particle-number sectors and retain whole spin multiplets. See
[physical conventions](../../docs/src/design/physical_conventions.md#hubbard-model)
and [Born sampling](../../docs/src/design/born_sampling.md).

## Parameters

All settings are defined in [config.jl](config.jl). The supplied configuration
uses the following parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `lattice` | `Lattice(2,2)` | Square lattice with four sites and open boundaries in both directions. |
| `t`, `U`, `mu` | 1, 8, 4 | Hopping, onsite repulsion and fixed chemical potential. |
| `symmetry` | `:u1su2` | Charge U(1) and spin SU(2), with fermion parity carried by the tensors. |
| `sampling_basis` | `SymmetryBasis()` | Draw complete empty, singly occupied and doubly occupied multiplets. |
| `Ns`, `seed` | 32, 20260924 | Initial Born sample count and random seed. |
| `D_pur`, `D_peps` | 8 each | Maximum PEPO preparation and PEPS evolution bond dimensions, including multiplet dimensions. |
| `chi_environment`, `chi_sampling`, `chi_measurement`, `chi_proposal` | 128 each | Boundary-MPS dimensions for full update, sampling, measurements and resampling proposals. |
| `delta_tau_pur`, `delta_tau_peps` | 0.01 each | Maximum imaginary-time increments for the thermal factor; delta_beta=2 delta_tau. |
| `trotter_order`, `update` | 2, `FullUpdate(...)` | Second-order Trotter evolution with environment-weighted tensor optimization. |
| `contraction` | `BoundaryMPSDiagonal` | Diagonal boundary-MPS contraction. |
| `resampling.enabled`, `resampling.eta_th` | `true`, 0.7 | Trigger annealed Born resampling when the sampling efficiency falls below 0.7. |

Temperatures are specified through inverse-temperature grids. With $k_B=1$
and $t=1$, $T=1/\beta$ in hopping units; beta=0 denotes infinite temperature.
The default grids are:

```julia
beta0 = 1.0
beta_log = collect(0.0:0.05:16.0)
beta_measure = collect(0.0:0.1:16.0)
```

Preparation reaches `beta0`; sampling and PEPS evolution continue to beta=16.
`beta_log` sets normalization, efficiency and resampling nodes, while
`beta_measure` selects energy measurements. The evolution uses the union of
these grids, 0 and `beta0`.

Start with the shared [TFIM quick start](../../README.md#a-quick-start) to
check the installation, and follow
[convergence](../../docs/src/guide/validation.md) when choosing Hubbard sample
counts and bond dimensions. Copy `config.jl` to `config.local.jl` for personal
runs; see [local configuration](../../docs/src/guide/quickstart.md#application-configuration).

## Tasks and output

Use `--stage=purification` to prepare the thermal PEPO at `beta0`, followed by
`--stage=stochastic` to draw Born samples and evolve them to lower
temperatures. The shared [stage table](../../docs/src/design/runtime.md#stages)
describes both entry points and checkpoint reuse.

### Run

Follow [environment and MPI setup](../../docs/src/guide/quickstart.md#application-environments-and-mpi)
with `APP=examples/Hubbard_model`, then run from the repository root:

```sh
julia --project=examples/Hubbard_model examples/Hubbard_model/main.jl --stage=purification
mpiexecjl --project=examples/Hubbard_model -n 2 julia --project=examples/Hubbard_model examples/Hubbard_model/main.jl --stage=stochastic
```

Use `-n 1` for one MPI rank. The [launch guide](../../docs/src/guide/quickstart.md#cluster-jobs)
covers Slurm submission.

Results are written under `output_directory`, which defaults to this example
directory in `config.jl`. `results/checkpoint.jld2` stores the thermal PEPO at
`beta0`, and `results/result.jld2` stores numerical curves, sample records and
diagnostics. `log/run.log` records progress and `log/samples/` contains sample
trajectories.

The thermal curve contains physical Hubbard energy per site with its sampling
error, log grand partition function and sampling efficiency. Sample records
also store particle number. See [reading results](../../docs/src/guide/results.md)
for completion checks and field definitions.

## Related guides

[Applications](../../docs/src/guide/applications.md),
[parameters](../../docs/src/guide/parameters.md),
[resampling](../../docs/src/design/resampling.md),
[results](../../docs/src/guide/results.md),
[convergence](../../docs/src/guide/validation.md),
[troubleshooting](../../docs/src/guide/troubleshooting.md).
