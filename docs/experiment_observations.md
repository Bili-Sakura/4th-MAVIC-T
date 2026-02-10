# Experiment Observations & Notes

> Record of observations and thinking from model training experiments.
> Add entries as experiments progress.

---

## Stage 1 — DDBM Latent Baseline vs CUT Ablation

### Order of Experiments

1. **DDBM latent-space baseline** (see [roadmap § Stage 1 — Latent-Space Modelling](roadmap.md#stage-1--latent-space-modelling-with-pre-trained-vae-baseline))
2. **CUT ablation** (see [roadmap § CUT Ablation (Stage 1)](roadmap.md#cut-ablation-stage-1))

---

### Experiment 1: DDBM Latent Baseline (DDBMLatentPipeline)

**Config**: Frozen pre-trained VAE (FLUX2-VAE / SD21-VAE), DDBM in latent space.

| Task      | Latent shape     | DDBM tier | ~Params |
|-----------|------------------|-----------|---------|
| RGB → IR  | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → IR  | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → RGB | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → EO  | `(32, 32, 32)`   | small     | ~20 M   |

**Observation**: Pure noise even after 5000 steps of training.

**Assumptions** (to investigate):

- Latent space of pre-trained RGB VAE may not align well with IR/SAR/EO modalities despite channel-repeat / channel-average.
<!-- - DDBM bridge formulation or training schedule may need tuning for remote-sensing latent distributions.
- Learning rate, batch size, or other hyperparameters may need adjustment. -->

---

### Experiment 2: CUT Ablation (Stage 1)

**Config**: GAN-based, pixel-space, no VAE. 512 px crop for 1024 px tasks.

| Task      | Pixel shape    | CUT tier | ~Params |
|-----------|----------------|----------|---------|
| RGB → IR  | `(512, 512)`   | large    | ~56.5 M |
| SAR → IR  | `(512, 512)`   | large    | ~56.5 M |
| SAR → RGB | `(512, 512)`   | large    | ~56.5 M |
| SAR → EO  | `(256, 256)`   | medium   | ~14.1 M |

**Observation**: Appealing results even at 1000 steps.

**Takeaways**:

- Pixel-space CUT converges much faster than latent-space DDBM in this setup.
- Fewer parameters (~56.5 M vs ~120 M) but better practical performance at early steps.
- Good candidate for quick iteration and baseline comparisons.

---

## Summary

| Approach          | Steps tested | Result              |
|-------------------|--------------|---------------------|
| DDBM latent       | 5000         | Pure noise          |
| CUT pixel-space   | 1000         | Appealing results   |

---

## Future Notes

Add new observations as experiments continue.
