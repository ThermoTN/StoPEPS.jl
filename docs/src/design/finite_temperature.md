# Finite-temperature evolution and weights

StoPEPS first prepares a purified thermal PEPO, then projects and evolves an
ensemble of PEPS samples. The method follows
[Gao et al., 2026](https://arxiv.org/abs/2609.35754); the
[notation guide](paper.md) maps its variables to the API.

## Purification

Starting from the identity PEPO, imaginary-time evolution gives 
$\rho(\beta/2) = e^{-\beta H/2} / \sqrt{Z}$. The checkpoint stores PEPO and its
global logarithmic amplitude.

## Stochastic evolution

Equations (5)–(7) and (28)–(29) of the paper evolve a normalized Born sample
from beta0:

```math
|\tilde\psi_k(\beta)\rangle
=e^{-(\beta-\beta_0)H/2}|\psi_k(\beta_0)\rangle,
\qquad w_k(\beta)=\|\tilde\psi_k(\beta)\|^2,
```

The normalized evolved sample is $|\psi_k(\beta)\rangle=|\tilde\psi_k(\beta)\rangle/\sqrt{w_k(\beta)}$.

```math
\langle O\rangle_\beta\simeq
\frac{\sum_k w_k\langle\psi_k(\beta)|O|\psi_k(\beta)\rangle}
{\sum_k w_k},\qquad
\ln Z(\beta)=\ln Z(\beta_0)+\ln\mathbb E[w_k(\beta)].
```

## Full update

Both PEPO preparation and PEPS evolution use two-site gates and full update,
following Appendix E.2. 
FU initialization and final-center SVD retry LAPACK nonconvergence once with
`gesvd` after `gesdd` fails, emitting a Julia warning with the operation, bond,
sector and matrix dimensions. Other failures propagate.

Boundary-MPS contractions supply the network environment
([Jordan et al., 2008](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.250602),
[Orús, 2014](https://doi.org/10.1016/j.aop.2014.06.013)), matching the references
used in Appendices E.1 and E.2 of the paper.
The [parameter reference](../guide/parameters.md) lists the solver and contraction
controls, and [results](../guide/results.md) defines their diagnostics.

## Temperature grid and normalization

Purification uses [temperature nodes](../guide/parameters.md)
through beta0; stochastic evolution uses nodes from
beta0 to the largest requested beta. `delta_tau_pur` and `delta_tau_peps` set
maximum factor-time increments, with steps shortened to reach output nodes.
Energy is measured at `beta_measure` nodes.

The stored state represents the thermal factor together with a logarithmic
amplitude. For direct Born continuation, the saved absolute sample weight is
Z(beta0)*w_k. More generally, a projected sample with physical squared norm G
and sampling probability q starts with `log_weight=log(G)-log(q)`.
The actual finite-chi probability is recorded as described in Appendix E.1.
Subsequent normalization factors update the sample weights.

`count` records repeated draws or selections. With multiplicities m_k, the
energy is the ratio of sums weighted by m_k*exp(log_weight), and `lnZ` is the
logarithm of their mean weight. [Annealed Born resampling](resampling.md)
constructs the proposal for regenerated samples.
