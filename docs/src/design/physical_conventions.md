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

## Hubbard model

The physical Hamiltonian is Eq. (8) of the paper:

```math
H=-t\sum_{\langle i,j\rangle,\sigma}
  (c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.})
  +U\sum_i n_{i\uparrow}n_{i\downarrow}.
```

`Hubbard(t=1,U=8,mu=4,symmetry=:u1su2)` evolves with K=H-mu*N at fixed mu.
The reported energy is H, and `lnZ` is the log grand partition function
log Tr exp(-beta*K). On the bipartite square lattice, mu=U/2 gives half filling
on average. Setting t=1 expresses energies and temperatures in hopping units.

The local multiplets follow Sec. II B and Fig. 1(e):

| Configuration label | Occupancy C_z | Stored charge q=C_z-1 | Spin S | Fermion parity | Auxiliary dimension |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | -1 | 0 | 0 | 1 |
| 2 | 1 | 0 | 1/2 | 1 | 2 |
| 3 | 2 | 1 | 0 | 0 | 1 |

Born sampling retains each complete multiplet. A sample has particle number
`sum(configuration .- 1)`, and the ensemble spans particle-number sectors.
At beta=0 the local probabilities are 1/4, 1/2 and 1/4.
