# Manual resampling benchmark: arguments [sweeps=1] [output.jld2] [checkpoint.jld2].
# One BLAS thread, two warmups and five measured repetitions.
using StoPEPS, LinearAlgebra, Statistics, JLD2
BLAS.set_num_threads(1)
sweeps=isempty(ARGS) ? 1 : parse(Int,ARGS[1])
options=ResamplingOptions(eta_th=1.,max_resample_sweep=sweeps)
c=RunConfig(Ns=16,D_pur=2,D_peps=2,chi_environment=8,chi_sampling=8,chi_proposal=8,chi_measurement=8,
    beta0=.1,beta_log=[0.,.1,.2,.3],beta_measure=[0.,.1,.3],resampling=options)
checkpoint_path=length(ARGS)>=3 ? ARGS[3] : nothing
checkpoint=checkpoint_path!==nothing && isfile(checkpoint_path) ? load_checkpoint(checkpoint_path,c) : run_purification(c)
checkpoint_path===nothing || isfile(checkpoint_path) || save_checkpoint(checkpoint_path,checkpoint)
for _ in 1:2; run_stochastic(c;checkpoint=checkpoint); end
runs=[];result=nothing
for _ in 1:5
    GC.gc()
    trial=@timed run_stochastic(c;checkpoint=checkpoint)
    global result=trial.value
    push!(runs,(;seconds=trial.time,bytes=trial.bytes,
        (;(k=>sum(getproperty(e.timing,k) for e in result["diagnostics"]["events"])
            for k in (:loo_seconds,:proposal_seconds,:replay_seconds,:total_seconds))...)...))
end
pick(x,ks)=(;(k=>getproperty(x,k) for k in ks)...)
particle(x)=pick(x,(:id,:count,:configuration,:log_p0,:log_q_proposal,:log_weight,:physical_log_norm,:energy_per_site))
stats(x)=pick(x,(:beta,:energy_per_site,:energy_per_site_error,:lnZ,:lnZ_error,:eta,:eta_error))
scientific=(;curve=stats.(result["observables"]["curve"]),samples=particle.(result["observables"]["samples"]),
    events=[(;pre=particle.(e.pre),fresh=particle.(e.fresh),post=particle.(e.post),
        selected=e.selection.selected_candidate,common_log_weight=e.selection.common_log_weight,
        lambdas=[x.plan.selection.lambda for x in e.lambda_evaluations],
        draws=[pick(d,(:configuration,:log_probability)) for d in [d for s in e.sweeps for d in s.draws]])
        for e in result["diagnostics"]["events"]])
benchmark=(;(k=>median(getproperty.(runs,k)) for k in keys(first(runs)))...)
println((;sweeps,benchmark...))
events=result["diagnostics"]["events"]
println((;completed_sweeps=[length(e.sweeps) for e in events],
    scoring_calls=sum(count(s->s.source==:score,e.score_table.entries) for e in events),
    scores_from_draws=sum(count(s->s.source==:draw,e.score_table.entries) for e in events)))
length(ARGS)>=2 && jldsave(ARGS[2];scientific,benchmark)
