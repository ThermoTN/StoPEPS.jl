using StoPEPS, LinearAlgebra
const config_path = let local_path = joinpath(@__DIR__, "config.local.jl")
    isfile(local_path) ? local_path : joinpath(@__DIR__, "config.jl")
end
include(config_path)
BLAS.set_num_threads(blas_threads)

stage_args = filter(a->startswith(a,"--stage="),ARGS)
length(stage_args)==1 || error("specify --stage=purification or --stage=stochastic")
all(a->startswith(a,"--stage=") || a=="--overwrite",ARGS) || error("unknown argument")
stage = split(only(stage_args),'=';limit=2)[2]
if stage=="purification"
    run_purification(config)
elseif stage=="stochastic"
    using MPI
    MPI.Init()
    try
        run_stochastic(config;comm=MPI.COMM_WORLD)
        MPI.Finalize()
    catch exception
        showerror(stderr,exception,catch_backtrace());println(stderr)
        MPI.Abort(MPI.COMM_WORLD,1)
    end
else
    error("unknown stage: $stage")
end
