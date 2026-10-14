# Perfect Born sampling

Perfect Born sampling constructs samples by projecting auxiliary indices of
the purified thermal supervector
([Gao et al., 2026](https://arxiv.org/abs/2609.35754), Sec. II and App. E.1):

```math
|\Psi(\beta_0)\rangle\!\rangle
=\frac{e^{-\beta_0 H/2}}{\sqrt{Z(\beta_0)}}|I\rangle\!\rangle,
\qquad
p_{\beta_0}(\alpha)=\|\langle\alpha|\Psi(\beta_0)\rangle\!\rangle\|^2.
```

The normalized projection is the Born sample. Its auxiliary configuration
is drawn sequentially from conditional probabilities, whose product gives
the full sampling probability. `SamplingCache` constructs the boundary-MPS
environments; `sample_configuration` draws a configuration and returns its
conditionals and `log_probability`. Passing `configuration` evaluates the
probability of a specified configuration using the same contractions.

Appendix E.1 describes the finite-chi approximation of these probabilities.
Increasing `chi_sampling` converges the boundary-MPS contraction. The saved
probability is the product of the conditionals used by the selected row or
diagonal traversal.

## Basis and symmetry

For dense TFIM, `XBasis()`, `YBasis()`, `ZBasis()` and `OrthonormalBasis(U)`
define local orthonormal bases; columns of U are physical kets. A matrix of
basis descriptors selects a basis at each site. A run uses one fixed choice
for initial sampling and resampling. The default `:auto` selects X for dense
TFIM and `SymmetryBasis()` for symmetric models.

Z₂ sampling selects parity eigenstates.

`project_configuration` applies the auxiliary projection and returns a PEPS
with its normalization records. For complex bases, auxiliary projection uses
the conjugate-basis bra so that the projected physical state is X|alpha>.
Physical proposals use the corresponding physical bra.

## Samples and weights

The Born probability is represented by the sample frequency, as in Eq. (4).
Further imaginary-time evolution contributes the squared-norm weights of
Eqs. (5)–(7). The implementation records probabilities and norms in logarithms;
the [thermal-weight guide](finite_temperature.md) gives their relation to
saved ensemble estimates.

Initial draws use `sample_rng(seed,sample_index)`; resampling streams depend
on the seed, event and draw indices. Numerical records identify the
random-stream conventions. The driver groups repeated configurations from
one sampling batch and stores their multiplicity as `count`.
The [library tutorial](../guide/library.md) shows an independent draw.
