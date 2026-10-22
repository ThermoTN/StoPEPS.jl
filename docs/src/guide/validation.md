# Convergence and uncertainty

Establish accuracy for the observable and temperature range of interest.
Keep the lattice, ensemble and energy units fixed when comparing runs, and
save each calculation in its own directory.

## Recommended sequence

1. **Compare a small independent reference.** Use `exact_thermal_observables`
   for a separate 2 × 2 TFIM calculation, as in the [library tutorial](library.md).
   Match the Ising spin/Pauli units to the
   [physical conventions](../design/physical_conventions.md).
2. **Converge state bond dimensions.** Increase `D_pur` for preparation, then
   `D_peps` for sample evolution. Check the energy and full-update residuals.
   Prepare a new checkpoint when studying purification accuracy.
3. **Converge boundary contractions.** Increase `chi_environment`,
   `chi_sampling`, `chi_measurement` and, with resampling, `chi_proposal`.
   Assess each contraction method at its chosen chi. Appendix E.1 of
   [the paper](../design/paper.md) discusses finite-chi sampling probabilities.
4. **Refine the time steps.** Reduce `delta_tau_pur` and `delta_tau_peps`
   while retaining the measurement endpoints. Tighten full-update tolerances
   and increase iteration budgets as needed. Compare direct continuation
   before checking the adaptive resampling calculation.
5. **Measure sampling uncertainty.** Use independent seeds at fixed numerical
   settings, then increase Ns. Reuse a thermal checkpoint to assess stochastic
   variation. Compare sample means, efficiency and their variation across runs.
6. **Check resampling effects.** Compare enabled resampling with direct
   continuation at matched numerical settings over independent seeds.
   Inspect eta and event histories.

## Statistical errors

`energy_per_site_error` estimates sampling uncertainty using the saved weights
and multiplicities. `lnZ_error` and `eta_error` are paired delta-method
estimates. Assess finite-D, finite-chi, Trotter and finite-size effects through
convergence studies, following the stoPEPS discussion in Appendix E.

For R independent runs with energy estimates E_r, report the mean and its
across-run standard error:

```math
\bar E=\frac1R\sum_r E_r,\qquad
\mathrm{SE}(\bar E)=\sqrt{\frac{\sum_r(E_r-\bar E)^2}{R(R-1)}}.
```

The across-run standard deviation describes the spread of individual runs;
dividing by sqrt(R) gives the standard error of their mean. Report statistical
errors and convergence shifts separately.

## Package and performance checks

`julia --project=. -e 'using Pkg; Pkg.test()'` checks package loading and the
default configuration. The manual tools `test/benchmark_resampling.jl` and
`test/benchmark_variational.jl` measure their respective calculation costs.
For MPI comparisons, use the same small configuration and seeds with one and
two ranks, choosing separate output directories. Compare numerical curves;
timings and rank ownership describe each execution.
