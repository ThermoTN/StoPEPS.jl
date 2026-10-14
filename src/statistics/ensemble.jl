"""
    ensemble_statistics(relative, multiplicity, energy, Ns; log_scale=0.0, nsites=1)

Weighted observables and paired delta-method sampling errors. `relative`
contains log weights relative to `log_scale`; `energy` contains energies per
site. Multiplicities count repeated samples in the ensemble.
"""
function ensemble_statistics(relative,multiplicity,energy,Ns;log_scale=0.0,nsites=1)
    length(relative)==length(multiplicity)==length(energy)>0 || error("empty or inconsistent ensemble")
    all(isfinite,relative) && all(m->m>0,multiplicity) || error("invalid particle weights/counts")
    Nv=sum(multiplicity); pivot=maximum(relative); w=exp.(relative.-pivot)
    mass=multiplicity.*w; total=sum(mass)
    mu1=total/Nv; mu2=sum(multiplicity.*w.^2)/Nv
    eta=clamp(mu1^2/mu2,1/Nv,1.0)
    sem(f)=Nv<=1 ? NaN : sqrt(max(0.0,sum(multiplicity.*abs2.(f))/(Nv*(Nv-1))))
    measured=all(isfinite,energy)
    E=measured ? sum(mass.*energy)/total : NaN
    Eerror=measured ? sem(w.*(energy.-E)./mu1) : NaN
    lnZ_error=sem((w.-mu1)./mu1)
    eta_error=sem((2mu1/mu2).*(w.-mu1).-(mu1^2/mu2^2).*(w.^2 .-mu2))
    return (;energy_total=nsites*E,energy_per_site=E,energy_per_site_error=Eerror,
        lnZ=log_scale+pivot+log(mu1),lnZ_error,eta,eta_error,ESS=Nv*eta,
        eta_p=clamp(total^2/(Ns*sum(abs2,mass)),1/Ns,length(relative)/Ns),
        target_samples=Ns,virtual_samples=Nv,representatives=length(relative))
end
function _aligned_weights(records)
    scale=maximum(r.weight_log_scale for r in records)
    relative=[r.relative_log_weight+(r.weight_log_scale-scale) for r in records]
    relative,scale
end
function _statistics(records,Ns,nsites)
    relative,scale=_aligned_weights(records)
    s=ensemble_statistics(relative,getfield.(records,:count),getfield.(records,:energy_per_site),Ns;
        log_scale=scale,nsites)
    return merge(s,(;beta=first(records).beta,
        max_state_truncation=maximum(r.max_state_truncation for r in records),
        max_environment_truncation=maximum(r.max_environment_truncation for r in records),
        max_boundary_entropy=maximum(r.max_boundary_entropy for r in records),
        distinct_configurations=length(Set(Tuple(vec(r.configuration)) for r in records))))
end
