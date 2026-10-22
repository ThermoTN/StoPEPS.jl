# Physical and lattice conventions

## Lattice and units

`Lattice(Lx,Ly)` is a rectangular square lattice with open boundaries.
Coordinates are `sites[x,y]`, with x increasing west to east and y increasing
north to south. The row-major site index is `x+(y-1)*Lx`; each nearest-neighbor
bond is counted once.

The code uses $k_B=1$. Thermal preparation evolves to $\rho(\beta/2)$.
`energy_per_site` is energy divided by Lx*Ly; `energy_total` is extensive.

## Transverse-field Ising model



The implementation `TFIM(J,h)` uses Pauli operators with eigenvalues ±1.

```math
H=-J\sum_{\langle ij\rangle} Z_i Z_j-h\sum_i X_i.
```

