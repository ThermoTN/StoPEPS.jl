# StoPEPS.jl

StoPEPS computes finite-temperature observables on open two-dimensional
lattices by combining PEPO purification, perfect Born sampling and PEPS
full-update evolution. The method is described by
[Gao et al., 2026](https://arxiv.org/abs/2609.35754).

## User guide

Start with the [4 × 4 TFIM calculation](https://github.com/ThermoTN/StoPEPS.jl#a-quick-start), then
[read its results](guide/results.md). Choose an [application](guide/applications.md)
and use [setup and launch](guide/quickstart.md) for local configuration and MPI.
Check the [parameters](guide/parameters.md) before increasing the system size;
[convergence](guide/validation.md) and [troubleshooting](guide/troubleshooting.md)
cover accuracy and calculation issues.

## Method and conventions

The [README Methods](https://github.com/ThermoTN/StoPEPS.jl#methods) gives a
short overview. These guides develop the equations, conventions and references:

- [Paper and notation](design/paper.md)
- [Physical and lattice conventions](design/physical_conventions.md)
- [Tensor and symmetry conventions](design/tensor_conventions.md)
- [Born sampling](design/born_sampling.md)
- [Thermal evolution and weights](design/finite_temperature.md)
- [Annealed Born resampling](design/resampling.md)

## Library and development

The [library tutorials](guide/library.md) show a serial calculation, an
independent Born draw and an exact-diagonalization comparison. The
[API reference](api.md) describes the public interfaces, and
[execution and output](design/runtime.md) covers stages and file handling.
