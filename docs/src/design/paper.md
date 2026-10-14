# Paper and notation

The stoPEPS method combines purification, symmetry-preserving Born sampling
and weighted imaginary-time evolution
([Gao et al., 2026](https://arxiv.org/abs/2609.35754)). Section and equation
numbers below refer to [version 1](https://arxiv.org/html/2609.35754v1).

## Notation

| Quantity | Paper notation | API |
|---|---|---|
| Number of lattice sites | ``N=L_xL_y`` | `length(config.lattice)` |
| Born sampling inverse temperature | ``\beta_0`` | `beta0` |
| Number of Born samples | ``N_{\rm s}`` | `Ns` |
| PEPO/PEPS bond dimension | ``D`` | `D_pur`, `D_peps` |
| Boundary-MPS bond dimension | ``\chi`` | `chi_environment`, `chi_sampling`, `chi_measurement`, `chi_proposal` |
| Auxiliary configuration | ``\alpha`` | `configuration` |
| Born sample | ``\lvert\psi_\alpha\rangle`` | `FinitePEPS` |
| Sampling probability | ``p_{\beta_0}(\alpha)`` | `exp(log_probability)` in a draw; `log_p0` in sample records |
| Sample weight | ``w_k`` | `log_weight`, with the common partition normalization included |
| Energy per site | ``\epsilon=E/N`` | `energy_per_site` |
| Sampling efficiency | ``\eta`` | `eta` |
| Mixture proportion, App. B | ``\lambda`` | Selected event `lambda`; controls in `ResamplingOptions` |
| Local Hubbard charge and spin | ``(C_z,S)`` | U(1)×SU(2) sectors with stored charge `q=C_z-1` |
| Ising couplings, Eq. (14) | ``J,B`` with ``S=\sigma/2`` | `TFIM(J=J/4,h=B/2)` in the same energy units |

## Methods and implementation

Section II and Appendix A define [perfect Born sampling](born_sampling.md).
Section VI and Appendix E describe the stoPEPS construction, boundary-MPS
contraction and [full-update evolution](finite_temperature.md). Appendix B
presents [annealed Born resampling](resampling.md).

`BoundaryMPSRow` follows the row-by-row traversal of Appendix E. The package
also provides diagonal and variational boundary-MPS contractions, selected
through `RunConfig.contraction`. The Hubbard example applies the multiplet
sampling of Sec. II B to U(1)×SU(2)-symmetric PEPO and PEPS tensors.

PEPS represent the spatial lattice directly
([Verstraete and Cirac, 2004](https://arxiv.org/abs/cond-mat/0407066)).
Environment-weighted tensor optimization follows the finite-PEPS framework of
[Lubasch et al., 2014](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.064425).
For boundary-MPS contraction, Appendices E.1 and E.2 cite
[Jordan et al., 2008](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.250602)
and [Orús, 2014](https://doi.org/10.1016/j.aop.2014.06.013),
Refs. [99] and [98] of the paper.

The [physical conventions](physical_conventions.md) give the Hamiltonians
and unit conversions. [Thermal weights](finite_temperature.md) connect the
paper's normalized samples to the saved logarithmic weights.

## Reference

- J. Gao et al., *Perfect Born Sampling of Symmetric Thermal Tensor Network for Quantum Lattice Models*, [arXiv:2609.35754](https://arxiv.org/abs/2609.35754).
- F. Verstraete and J. I. Cirac, *Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions*, [arXiv:cond-mat/0407066 (2004)](https://arxiv.org/abs/cond-mat/0407066).
- M. Lubasch, J. I. Cirac and M.-C. Bañuls, *Algorithms for finite projected entangled pair states*, [Phys. Rev. B **90**, 064425 (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.064425).
- J. Jordan et al., *Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions*, [Phys. Rev. Lett. **101**, 250602 (2008)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.250602).
- R. Orús, *A practical introduction to tensor networks: Matrix product states and projected entangled pair states*, [Annals of Physics **349**, 117 (2014)](https://doi.org/10.1016/j.aop.2014.06.013).
