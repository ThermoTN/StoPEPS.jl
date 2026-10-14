# Physical and lattice conventions

## Lattice and units

`Lattice(Lx,Ly)` is a rectangular square lattice with open boundaries.
Coordinates are `sites[x,y]`, with x increasing west to east and y increasing
north to south. The row-major site index is `x+(y-1)*Lx`; each nearest-neighbor
bond is counted once.

The code uses k_B=1. Temperature has energy units; beta and imaginary time
have inverse-energy units. Thermal preparation evolves to tau=beta/2.
`energy_per_site` is energy divided by Lx*Ly; `energy_total` is extensive.

## Transverse-field Ising model

The Hamiltonian is Eq. (14) of [the paper](paper.md):

```math
H=-J\sum_{\langle i,j\rangle}S_i^zS_j^z-B\sum_iS_i^x.
```

The implementation `TFIM(J,h)` uses Pauli operators with eigenvalues ±1.

```math
H=-J\sum_{\langle ij\rangle} Z_i Z_j-h\sum_i X_i.
```

The paper uses spin operators S=Pauli/2. Its J=1, B=1.52 corresponds to
`TFIM(J=0.25,h=0.76)` in the same energy units. The shipped
`TFIM(J=1,h=3.04)` configuration uses four times that energy scale:
`T_paper=T/4` and `E_paper=E/4`.
