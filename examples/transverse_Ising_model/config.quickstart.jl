using StoPEPS

function quickstart_config(;
    lattice=Lattice(4, 4), # 4 × 4 square lattice with open boundaries
    output_directory=joinpath(@__DIR__, "output", "quickstart"),
    seed=20260924,
    overwrite_output=false,
)
    RunConfig(
        lattice=lattice,
        model=TFIM(J=1.0, h=0.7),
        sampling_basis=XBasis(),
        beta0=0.1,
        beta_log=[0.0, 0.1, 0.2, 0.3],
        beta_measure=[0.0, 0.1, 0.2, 0.3],
        delta_tau_pur=0.025,
        delta_tau_peps=0.025,
        D_pur=4,
        D_peps=4,
        chi_environment=16,
        chi_sampling=16,
        chi_measurement=16,
        chi_proposal=16,
        update=FullUpdate(max_iterations=40, linear_max_iterations=100),
        contraction=BoundaryMPSRow,
        Ns=32,
        seed=seed,
        resampling=ResamplingOptions(enabled=false),
        output_directory=output_directory,
        overwrite_output=overwrite_output,
    )
end
