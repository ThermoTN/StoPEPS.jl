# Tensor and symmetry conventions

## PEPO and PEPS tensors

`FinitePEPO` and `FinitePEPS` are `FiniteState` types with local TensorKit
tensors in outgoing order `(W,N,Pout,Paux,E,S)`. West/north virtual spaces are
dual to east/south spaces. Open boundary legs use unit spaces. Double layers
pair `(ket,bra)` and fuse to `(W,N,E,S)`.

PEPO physical and auxiliary legs represent the doubled Hilbert space in
Eq. (16) of [the paper](paper.md). A Born sample retains the selected auxiliary
multiplet: dimension one for TFIM and dimensions 1,2,1 for Hubbard's
empty/single/double outcomes. Measurements contract the auxiliary carrier.

TensorKit permutations, fusion maps, QR and SVD preserve the native spaces and
their orientations. Bond dimensions count the full representation dimension;
sector capacities additionally control the number of retained multiplets.
The state amplitude includes `exp(log_amplitude)`, which accumulates local
tensor normalization factors.

## Fermionic representation

Hubbard tensors use `U1Irrep ⊠ SU2Irrep ⊠ FermionParity`, with charge n-1
and parity n modulo 2. TensorKit braiding supplies fermionic signs. The
contraction kernels use evaluation-map twists to connect diagrammatic
contractions with the positive metrics used by QR, SVD and full update.
Norms and truncations include multiplet dimensions.

## Boundary MPS

`BoundaryMPSRow` absorbs rows and compresses their canonical boundary MPSs.
`BoundaryMPSDiagonal` absorbs layers of constant x+y. Its traversal increases
x on even layers and decreases x on odd layers. The variational methods fit
each absorbed row or diagonal using two-site updates.

The same contraction choice supplies full-update environments, observables,
Born probabilities and resampling proposals. Independent `chi_*` controls
set their bond dimensions. Configuration storage remains `sites[x,y]` for
each traversal. Saved `sampling_site_order` gives the conditional order.
See [parameters](../guide/parameters.md) for selecting a method and
[results](../guide/results.md) for its error and entropy diagnostics.
