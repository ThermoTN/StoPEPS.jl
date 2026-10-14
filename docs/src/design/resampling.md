# Annealed Born resampling

Annealed Born resampling constructs an improved proposal from the cooled
ensemble to restore sampling efficiency
([Gao et al., 2026](https://arxiv.org/abs/2609.35754), App. B). With sample
weights w_k, Eq. (21) defines

```math
\eta=\frac{(\sum_{k=1}^{N_{\rm s}}w_k)^2}
{N_{\rm s}\sum_{k=1}^{N_{\rm s}}w_k^2}.
```

The empirical proposal of Eq. (23) is

```math
P_\beta(\alpha;N_{\rm s})=
\frac{\sum_k w_k|\langle\alpha|\psi_k(\beta)\rangle|^2}
{\sum_k w_k}.
```

It is combined with the initial Born distribution:

```math
P_\beta^{\rm mix}(\alpha)=
\lambda P_\beta(\alpha;N_{\rm s})+(1-\lambda)p_{\beta_0}(\alpha).
```

The mixture proportion is selected by minimizing the second moment of the
resampled weights. A fresh auxiliary configuration is projected onto the
purified supervector at beta0, with relative initial weight
`p_beta0(alpha)/P_mix(alpha)`. Imaginary-time evolution carries the regenerated
sample to the current temperature, where it is combined with the parent ensemble
and selected using importance weights, following Algorithm 2.

The [parameter reference](../guide/parameters.md)
describes how to enable this procedure and choose its controls;
[results](../guide/results.md) describes the saved events.
