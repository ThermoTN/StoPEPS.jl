# Manual benchmark: julia --project=. test/benchmark_variational.jl [local|workflow]
# Fixed inputs, one BLAS thread, warmed median timings and allocated bytes.
using StoPEPS, TensorKit, TensorOperations, LinearAlgebra, Random, Statistics
const SP=StoPEPS
BLAS.set_num_threads(1)
# Self-contained benchmark input builders.
function random_state(l;aux=2,seed=41,D=2,T=ComplexF64)
    rng=MersenneTwister(seed);P=ComplexSpace(2);Q=ComplexSpace(aux)
    sites=[begin
        W=ComplexSpace(x==1 ? 1 : D);N=ComplexSpace(y==1 ? 1 : D)
        E=ComplexSpace(x==l.Lx ? 1 : D);S=ComplexSpace(y==l.Ly ? 1 : D)
        spaces=(W',N',P,Q',E,S)
        TensorMap(randn(rng,T,map(dim,spaces)...),SP._product(spaces),one(P))
    end for x in 1:l.Lx,y in 1:l.Ly]
    (aux==1 ? FinitePEPS : FinitePEPO)(l,sites)
end
function graded_random_state(lattice,physical,virtual;Scalar=ComplexF64,seed=618)
    rng=MersenneTwister(seed)
    sites=[begin
        west=xpos==1 ? oneunit(virtual) : virtual
        north=ypos==1 ? oneunit(virtual) : virtual
        east=xpos==lattice.Lx ? oneunit(virtual) : virtual
        south=ypos==lattice.Ly ? oneunit(virtual) : virtual
        randn(rng,Scalar,west'⊗north'⊗physical⊗physical'⊗east⊗south←one(physical))
    end for xpos in 1:lattice.Lx,ypos in 1:lattice.Ly]
    FinitePEPO(lattice,sites)
end
function replace_config(c;kwargs...)
    fields=(; (name=>getfield(c,name) for name in fieldnames(RunConfig))...)
    RunConfig(;merge(fields,(;kwargs...))...)
end
function bench(label,f;warm=2,reps=5)
    for _ in 1:warm; f(); end
    times=Float64[];bytes=Int[];value=nothing;events=Any[]
    for _ in 1:reps
        GC.gc(); r=@timed f();push!(times,r.time);push!(bytes,r.bytes);value=r.value
        if value isa AbstractDict && haskey(value,"diagnostics")
            push!(events,value["diagnostics"]["events"])
        end
    end
    println((label=label,seconds=median(times),MiB=median(bytes)/2.0^20));flush(stdout)
    if !isempty(events)
        println((label=label*" resample phases",medians=(;
            (k=>median([sum(getproperty(e.timing,k) for e in batch;init=0.) for batch in events])
                for k in (:loo_seconds,:proposal_seconds,:replay_seconds,:total_seconds))...)))
        flush(stdout)
    end
    value
end
if isempty(ARGS) || "local" in ARGS
for (name,P,V,D,chi,l) in (("Z2",Z2Space(0=>1,1=>1),Z2Space(0=>2,1=>2),4,32,Lattice(4,4)),
        ("Hubbard",physical_space(Hubbard()),physical_space(Hubbard())⊕physical_space(Hubbard()),8,128,Lattice(2,2)))
    state=graded_random_state(l,P,V;seed=618)
    for algorithm in (BoundaryMPSDiagonal,BoundaryMPSDiagonalVariational)
        method=algorithm(chi)
        result=bench("$name norm $algorithm",()->lognorm(state,method))
        records=hasproperty(result,:boundary_variational) ? result.boundary_variational.layers : []
        println((log_norm=result.log_norm,sweeps=[r.sweeps for r in records],
            layer_fallbacks=count(r->r.fallback!==nothing,records)));flush(stdout)
    end
    if hasfield(BoundaryMPSDiagonalVariational,:target_cache_bytes)
        input=SP.DiagonalBoundary();layer=name=="Z2" ? 5 : 3
        for t in 2:layer-1
            SP._diagonal_absorb_layer!(input,state.sites,l,t,BoundaryMPSDiagonal(chi))
        end
        target=SP._diagonal_layer_target(input,state.sites,l,layer)
        C,report,_=SP._variational_fit(target,BoundaryMPSDiagonalVariational(chi),nothing)
        println((audit=name,layer=layer,input_bonds=[dim(space(A,3)) for A in input.tensors],
            candidate_bonds=[dim(space(A,3)) for A in C.tensors],
            cache_peak_bytes=target.cache.peak_bytes,raw_builds=target.cache.builds,
            cache_hits=target.cache.hits,sweeps=report.sweeps,work=target.work));flush(stdout)
    end
end
for D in (2,3,4), chi in (4,8)
    state=random_state(Lattice(3,3);D,seed=419)
    input=SP.DiagonalBoundary();SP._diagonal_absorb_layer!(input,state.sites,state.lattice,2,BoundaryMPSDiagonal(chi))
    bench("dense layer D=$D chi=$chi",()->SP._diagonal_absorb_layer!(copy(input),state.sites,state.lattice,3,BoundaryMPSDiagonalVariational(chi)))
end
end
# Short, reproducible workflows use the example's lattice, model and capacity,
# with one purification step, four draws, and one forced resample/replay endpoint.
if isempty(ARGS) || "workflow" in ARGS
for (name,model,l,D,chi) in (("Z2",TFIM(h=3.04,physical_space=Z2Space(0=>1,1=>1)),Lattice(4,4),4,32),
        ("Hubbard",Hubbard(U=8,mu=4),Lattice(2,2),8,128))
    base=RunConfig(lattice=l,model=model,D_pur=D,D_peps=D,chi_environment=chi,chi_sampling=chi,chi_measurement=chi,chi_proposal=chi,
        beta0=.02,beta_log=[0.,.02,.04],beta_measure=[0.,.02,.04],delta_tau_pur=.01,delta_tau_peps=.01,Ns=4,seed=20260924,
        contraction=BoundaryMPSDiagonal,resampling=ResamplingOptions(eta_th=1.))
    cp=run_purification(base)
    for algorithm in (BoundaryMPSDiagonal,BoundaryMPSDiagonalVariational)
        cfg=replace_config(base;contraction=algorithm);method=algorithm(chi)
        bench("$name FU $algorithm",()->evolution_step!(copy(cp.state),model,.01,D,FullUpdate(),method);warm=1,reps=3)
        bench("$name measure $algorithm",()->measure_energy(cp.state,model,method);warm=1,reps=3)
        cache=bench("$name sampling-cache $algorithm",()->SamplingCache(cp.state,method);warm=1,reps=3)
        bench("$name Born $algorithm",()->sample_configuration(cache;rng=sample_rng(20260924,1));warm=1,reps=3)
        result=bench("$name trajectory $algorithm",()->run_stochastic(cfg;checkpoint=cp);warm=1,reps=3)
        events=result["diagnostics"]["events"]
        @assert !isempty(events) "benchmark must exercise forced resample/replay"
        println((workflow=name,algorithm=string(algorithm),events=length(events),curve=result["observables"]["curve"]));flush(stdout)
    end
end
end
