using StoPEPS, TensorKit

# Pauli convention: H = -J sum_<ij> Z_i Z_j - h sum_i X_i; finite OBC.
# The paper uses S=Pauli/2: these couplings give T_paper=T/4 and E_paper=E/4.
# Copy to config.local.jl for machine-specific controls (ignored by Git).
# Each stage loads config.local.jl when present, otherwise this file.
const blas_threads = 1
const config = RunConfig(
    lattice = Lattice(4, 4),
    model = TFIM(
        J=1.0,
        h=3.04,
        physical_space=Z2Space(0=>1,1=>1)
        ),
    # SymmetryBasis samples parity sectors; dense TFIM also accepts X/Y/Z bases.
    sampling_basis = SymmetryBasis(),
    beta0 = 1.0,
    beta_log = collect(0.0:0.05:8.0),
    beta_measure = collect(0.0:0.5:8.0),
    delta_tau_pur = 0.025,
    delta_tau_peps = 0.025,
    D_pur = 4,
    D_peps = 4,
    chi_environment = 32,
    chi_sampling = 32,
    chi_measurement = 32,
    chi_proposal = 32,
    trotter_order = 2,
    update = FullUpdate(
        # Stop when the environment-weighted normalized squared fit error
        # reaches error_tolerance, the same metric as FU truncation in the log.
        max_iterations=40, error_tolerance=1e-8,
        diagonal_shift=1e-10,
        linear_max_iterations=100, linear_tolerance=1e-9,
    ),
    # Row and variational contraction controls: docs/src/guide/parameters.md.
    contraction = BoundaryMPSDiagonal,
    record_boundary_entropy = true, # E.E. in nats; Se layout depends on contraction
    Ns = 64,
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
