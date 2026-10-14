# Architecture

StoPEPS implements finite-temperature stochastic PEPS with perfect Born
sampling and full-update imaginary-time evolution on finite open lattices.
The [documentation index](src/index.md) provides the user guides, and
[paper and notation](src/design/paper.md) connects the implementation to the paper.

| Source directory | Responsibility |
|---|---|
| `src/states/` | Finite PEPO/PEPS tensors and normalization |
| `src/models/` | Ising and Hubbard operators, gates and exact references |
| `src/contraction/` | Row and diagonal boundary-MPS contractions |
| `src/evolution/` | Two-site full update and environment repair |
| `src/sampling/` | Born sampling, basis definitions and projection |
| `src/resample/` | Mixture proposals and importance resampling |
| `src/measurement/` | Energy and particle-number measurements |
| `src/statistics/` | Weighted observables and sampling uncertainty |
| `src/driver/` | Configuration and the two calculation stages |
| `src/io/` | Logs, numerical results and the thermal checkpoint |
| `ext/StoPEPSMPIExt.jl` | Distributed samples and MPI collectives |

`FinitePEPO` and `FinitePEPS` share native TensorKit contraction and evolution
kernels. Their local legs and symmetry spaces are specified in
[tensor conventions](src/design/tensor_conventions.md).

`FullUpdate` provides two-site imaginary-time evolution. `ContractionMethod`
dispatch selects row, diagonal or variational boundary MPSs for norms,
observables and sampling. Models implement `physical_space`, local Hamiltonians
and energy measurement.

See [execution and output](src/design/runtime.md) for stage controls,
checkpoint reuse and the file layout, and
[thermal evolution](src/design/finite_temperature.md) for weights and observables.
