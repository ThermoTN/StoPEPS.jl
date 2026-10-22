# Troubleshooting

Start with the [4 × 4 calculation](quickstart.md). Completion metadata and
the last collected curve row identify the progress of each stage.

| Symptom | Checks and action |
|---|---|
| Package resolution error | Instantiate the project selected by `--project`. Application environments load this checkout through `[sources]`. |
| Launcher lookup | Install `mpiexecjl` in the application environment and add the depot's bin directory to PATH, or use its absolute path. |
| MPI initialization or communication error | Inspect `using MPI; MPI.versioninfo()` in the application environment. Use a matching launcher/backend, then check one rank and two ranks. |
| Existing checkpoint or output | Choose a fresh `output_directory`, or set `overwrite_output=true` to replace the calculation. The CLI accepts `--overwrite`. |
| Checkpoint mismatch | Compare the saved lattice, model, physical representation and beta0 with the current config. Prepare a matching checkpoint. |
| Local configuration error | Check that `config.local.jl` defines both `config` and `blas_threads`, and that all ranks read the same file. |
| Born conditional error | Inspect the reported site and increase `chi_sampling`; compare with a small reference and a larger preparation bond dimension. |
| FullUpdate iteration summary | Inspect the fit residual against `error_tolerance`; assess a larger `max_iterations` and a more accurate environment. |
| FullUpdate linear-solver summary | Check `linear_tolerance`, `linear_max_iterations` and `diagonal_shift` together with the final fit residual. |
| NaN energy | Check that the beta belongs to `beta_measure`. Norm and efficiency records are available at the other output nodes. |
| Zero lnZ or eta error after resampling | Equal weights can give zero conditional errors. Use independent-run dispersion and the [convergence guide](validation.md). |
| Small sampling efficiency | Increase Ns, assess beta0 and enable annealed Born resampling. Check convergence over independent seeds. |
| Long first call | Julia compilation can dominate a small calculation. Measure process wall time and evolution-step timings separately. |
| Memory pressure | Start with the quick-start dimensions and measure increasing D and chi. `target_cache_bytes` controls the diagonal variational target-block cache. |

The [parameter reference](parameters.md) defines the numerical controls.
The [results tutorial](results.md) describes progress, diagnostics and
optional measurements.
