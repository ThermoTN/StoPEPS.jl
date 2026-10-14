using StoPEPS, TensorKit

# Copy to config.local.jl for machine-specific controls (ignored by Git).
# Each stage loads config.local.jl when present, otherwise this file.
const blas_threads = 1
const config = RunConfig(
    lattice = Lattice(2, 2),
    model = Hubbard(t=1.0, U=8.0, mu=4.0, symmetry=:u1su2),
    sampling_basis = SymmetryBasis(),
    beta0 = 1.0,
    beta_log = collect(0.0:0.05:16.0),
    beta_measure = collect(0.0:0.1:16.0),
    delta_tau_pur = 0.01,
    delta_tau_peps = 0.01,
    D_pur = 8,
    D_peps = 8,
    chi_environment = 128,
    chi_sampling = 128,
    chi_measurement = 128,
    chi_proposal = 128,
    trotter_order = 2,
    update = FullUpdate(
        max_iterations=40, error_tolerance=1e-8,
        diagonal_shift=1e-10,
        linear_max_iterations=100, linear_tolerance=1e-9,
    ),
    # Row and variational contraction controls: docs/src/guide/parameters.md.
    contraction = BoundaryMPSDiagonal,
    record_boundary_entropy = true, # Includes SU(2) representation dimensions
    Ns = 32,
    seed = 20260924,
    resampling = ResamplingOptions(
        enabled=true, eta_th=0.7, lambda_max=0.99999,
        lambda_tolerance=1e-8,
        lambda_threshold=0.2, max_resample_sweep=3,
        score_batch_size=32,
    ),
    save_interval_seconds = 3600.0,
    output_directory = @__DIR__,
    overwrite_output = "--overwrite" in ARGS,
)
