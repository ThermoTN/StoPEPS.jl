# StoPEPS.jl

## Introduction

StoPEPS.jl is a Julia package for finite-temperature quantum many-body
calculations on 2D lattices. It implements
stochastic projected entangled pair states (stoPEPS), based on perfect Born
sampling of symmetric thermal tensor networks
([Gao et al., 2026](https://arxiv.org/abs/2609.35754)).

## Methods

The algorithm combines thermal purification at high temperatures with
stochastic sampling at low temperatures. PEPS represent the 2D
lattice directly, following
[Verstraete and Cirac, 2004](https://arxiv.org/abs/cond-mat/0407066).

### Purification

We first represent
$\rho(\beta/2) = e^{-\beta H/2}/\sqrt{Z}$
as a projected entangled pair operator (PEPO). Starting from the identity at infinite temperature,
imaginary-time evolution reaches the sampling inverse temperature `beta0`.
Two-site Trotter gates are compressed by full update
([Lubasch et al., 2014](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.064425)).
Boundary matrix product states (MPS) contract the surrounding network
([Jordan et al., 2008](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.250602),
[Orús, 2014](https://doi.org/10.1016/j.aop.2014.06.013)).


### Stochastic sampling

At `beta0`, perfect Born sampling sequentially draws auxiliary configurations
from the purified PEPO. Each sample is then
evolved to lower temperatures by full update. Observables are estimated as
ratios of weighted sums, with weights accounting for the sampling probability
and subsequent imaginary-time evolution. The boundary-MPS dimension controls
the contraction accuracy of the Born probabilities and measurements.


## A quick start

Run these commands from the repository root with Julia 1.11 or later:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e '
using StoPEPS, LinearAlgebra
BLAS.set_num_threads(1)
include("examples/transverse_Ising_model/config.quickstart.jl")
c = quickstart_config()
checkpoint = run_purification(c)
run_stochastic(c; checkpoint)
'
```

The [quick-start configuration](examples/transverse_Ising_model/config.quickstart.jl)
runs a 4 × 4 transverse-field Ising model with open boundaries, J=1 and h=0.7,
in one process with one BLAS thread. Output goes to the ignored directory
`examples/transverse_Ising_model/output/quickstart/`. A successful run has four
curve rows and `metadata["complete"]==true`; the
[results tutorial](docs/src/guide/results.md) checks these and prints energy
per site with its sampling error. See [setup and launch](docs/src/guide/quickstart.md)
for the example settings, local configuration and MPI setup.

If the output directory already contains a run, choose a fresh directory with
`quickstart_config(output_directory="output/quickstart-run2")`, or set
`overwrite_output=true` to replace the previous run.

## Examples

The [transverse-field Ising example](examples/transverse_Ising_model/) computes thermal observables on square
lattices with open boundaries. 

The example README gives its Hamiltonian, parameter table, temperature grid
and output files. See the [application guide](docs/src/guide/applications.md)
and the [physical conventions](docs/src/design/physical_conventions.md)
for energy units and symmetry labels.

## Documentation and development

The [manual](docs/src/index.md) links the user guides, method descriptions and
API reference. Follow [convergence and uncertainty](docs/src/guide/validation.md)
when choosing sample counts, bond dimensions and time steps.

TensorKit supplies symmetric tensor operations; StoPEPS supplies full update,
boundary-MPS contractions, Born sampling and the serial/MPI driver. The
library supports Julia 1.10 or later; examples and documentation use Julia
1.11 or later. MPI is optional for library calculations. The
[architecture](docs/architecture.md) and [result format](docs/result-schema.md)
describe the implementation and saved data.

The package-loading and default-configuration smoke check is
`julia --project=. -e 'using Pkg; Pkg.test()'`. Small-system comparisons are
shown in the [library tutorials](docs/src/guide/library.md).

Build the manual with
`julia --project=docs -e 'using Pkg; Pkg.instantiate()'` followed by
`julia --project=docs docs/make.jl`.


## Maintainer

- Jianxin Gao, <jxgao@buaa.edu.cn>, Beihang University
- Chuanshu Xu, <xuchuanshu@itp.ac.cn>, ITP-CAS



## Reference

- J. Gao et al., *Perfect Born Sampling of Symmetric Thermal Tensor Network for Quantum Lattice Models*, [arXiv:2609.35754](https://arxiv.org/abs/2609.35754).
- F. Verstraete and J. I. Cirac, *Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions*, [arXiv:cond-mat/0407066 (2004)](https://arxiv.org/abs/cond-mat/0407066).
- M. Lubasch, J. I. Cirac and M.-C. Bañuls, *Algorithms for finite projected entangled pair states*, [Phys. Rev. B **90**, 064425 (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.064425).
- J. Jordan et al., *Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions*, [Phys. Rev. Lett. **101**, 250602 (2008)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.250602).
- R. Orús, *A practical introduction to tensor networks: Matrix product states and projected entangled pair states*, [Annals of Physics **349**, 117 (2014)](https://doi.org/10.1016/j.aop.2014.06.013).
