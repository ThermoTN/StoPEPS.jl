# Parameter reference

`RunConfig` holds the model, numerical settings and output controls.
`validate_config(config)` checks their consistency. Tables below give library
defaults; the application `config.jl` files and quick start supply their own
explicit settings.

## Model and temperature

| Parameter | Default | Meaning |
|---|---|---|
| `lattice` | `Lattice(2,2)` | Positive Lx and Ly; square lattice with open boundaries |
| `model` | `TFIM(J=1,h=0.7)` | Pauli Ising Hamiltonian |
| `beta0` | 0.1 | Born sampling inverse temperature |
| `beta_log` | 0:0.05:0.3 | Output nodes for normalization, efficiency and resampling |
| `beta_measure` | [0,0.1,0.2,0.3] | Energy measurement nodes |
| `delta_tau_pur`, `delta_tau_peps` | 0.025 | Positive factor-time steps; delta_beta=2 delta_tau |
| `trotter_order` | 2 | First- or second-order imaginary-time evolution |

Beta values are finite and nonnegative. The resolved grid is the sorted union
of both vectors, 0 and beta0, ending at its largest value. The
[physical conventions](../design/physical_conventions.md) give the Ising
operator convention and units.

## Tensor dimensions and contraction

| Parameter | Default | Meaning |
|---|---|---|
| `D_pur`, `D_peps` | 4 | Positive PEPO preparation and PEPS evolution bond dimensions |
| `chi_environment` | 32 | Full-update boundary-MPS dimension |
| `chi_sampling` | chi_environment | Initial Born sampling boundary dimension |
| `chi_measurement` | chi_environment | Norm and energy contraction dimension |
| `chi_proposal` | chi_sampling | Resampling proposal boundary dimension |
| `bond_space_pur`, `bond_space_peps` | nothing | Optional TensorKit sector capacities for state bonds |
| `environment_space` | nothing | Optional sector capacity for full-update boundaries |
| `sampling_space`, `measurement_space` | environment_space | Sampling and measurement boundary sector capacities |
| `proposal_space` | sampling_space | Proposal boundary sector capacity |
| `contraction` | BoundaryMPSRow | Contraction type described below |
| `max_sweep` | 4 | Complete round trips for variational boundary fits |
| `threshold_sweep` | 1e-8 | Relative squared-norm change, compared from round trip two; zero runs the full sweep budget |
| `target_cache_bytes` | 268435456 | Cached target-block payload per active diagonal variational layer; zero recomputes blocks |
| `record_boundary_entropy` | true | Record boundary-MPS entropy in nats |

Integer dimensions count complete representation dimensions. Optional sector
spaces share the model symmetry and use nondual spaces with positive dimension.
Truncation applies the integer dimension and sector capacity together.

Choose `BoundaryMPSRow` (`BoundaryMPS`), `BoundaryMPSDiagonal`,
`BoundaryMPSRowVariational` or `BoundaryMPSDiagonalVariational`. The
[tensor conventions](../design/tensor_conventions.md) describe
their traversals; [reading results](results.md)
defines their contraction diagnostics. Assess each method's chi convergence
for the observable of interest.

## Full-update solver

`update=FullUpdate(...)` configures environment-weighted tensor optimization
with [external and internal gauges](../design/finite_temperature.md#full-update).

| Keyword | Library default | Meaning |
|---|---|---|
| `max_iterations` | 4 | Alternating least-squares iterations; zero selects the initial environment-gauged SVD fit |
| `error_tolerance` | 1e-8 | Relative squared fit residual of paper Eq. (34) |
| `diagonal_shift` | 1e-10 | Relative regularization shift in the current gauged core linear system |
| `linear_max_iterations` | 24 | Preconditioned conjugate-gradient iterations |
| `linear_tolerance` | 1e-9 | Relative linear-system residual |

The shipped configurations use 40 alternating iterations and 100 linear
iterations. Inspect the fit residual and solver diagnostics when choosing
these budgets.

## Sampling and resampling

| Parameter | Library default | Meaning |
|---|---|---|
| `sampling_basis` | :auto | X basis for dense TFIM; symmetry multiplets for symmetric models |
| `Ns` | 16 | Positive initial sample count; enabled resampling uses Ns>=2 |
| `seed` | 20260924 | Nonnegative sampling seed |
| `resampling` | ResamplingOptions() | Annealed Born resampling controls |

See [Born sampling](../design/born_sampling.md) for basis
descriptors and symmetry multiplets.

`ResamplingOptions(enabled=true,eta_th=0.7)` triggers resampling at output
nodes with eta below the threshold. `enabled=false` selects direct evolution
of the initial Born samples. The shipped configs additionally set
`lambda_max=0.99999`, `lambda_tolerance=1e-8`, `lambda_threshold=0.2`,
`max_resample_sweep=3` and `score_batch_size=32`; the library's `lambda_max`
default is 0.999. These control the mixture search, fresh sampling batches
and probability evaluation. See [annealed Born resampling](../design/resampling.md).

## Execution and output

| Parameter | Default | Meaning |
|---|---|---|
| `output_directory` | nothing | In-memory library calculation; a path enables logs and JLD2 files |
| `overwrite_output` | false | Set true to replace existing output |
| `save_interval_seconds` | 3600 | Periodic numerical writes at safe calculation boundaries; zero writes at each opportunity |
| `blas_threads` | 1 in examples | BLAS threads per process, set by the example entry point |

See [execution and output](../design/runtime.md) for stage controls and file
handling. Use a fresh directory for each convergence or independent-seed
calculation.
