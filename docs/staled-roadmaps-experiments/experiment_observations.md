# Experiment Observations & Notes

> Record of observations and thinking from model training experiments.
> Add entries as experiments progress.


---

### Experiment 1: DDBM Latent Baseline (DDBMLatentPipeline)

**DDBM latent-space baseline** (see [roadmap § Stage 1 — Latent-Space Modelling](roadmap.md#stage-1--latent-space-modelling-with-pre-trained-vae-baseline))

**Config**: Frozen pre-trained VAE (FLUX2-VAE / SD21-VAE), DDBM in latent space.

| Task      | Latent shape     | DDBM tier | ~Params |
|-----------|------------------|-----------|---------|
| RGB → IR  | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → IR  | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → RGB | `(128, 128, 32)` | medium    | ~120 M  |
| SAR → EO  | `(32, 32, 32)`   | small     | ~20 M   |

**Observation**: Pure noise even after 5000 steps of training. **Status**: Failed.

**Checkpoint**: <https://huggingface.co/BiliSakura/4th-MAVIC-T-ckpt-failed>

**Assumptions** (to investigate):

- Latent space of pre-trained RGB VAE may not align well with IR/SAR/EO modalities despite channel-repeat / channel-average.
<!-- - DDBM bridge formulation or training schedule may need tuning for remote-sensing latent distributions.
- Learning rate, batch size, or other hyperparameters may need adjustment. -->

---

### Experiment 2: CUT Ablation (Stage 1)

**CUT ablation** (see [roadmap § CUT Ablation (Stage 1)](roadmap.md#cut-ablation-stage-1))
**Config**: GAN-based, pixel-space, no VAE. 512 px crop for 1024 px tasks.

| Task      | Pixel shape    | CUT tier | ~Params |
|-----------|----------------|----------|---------|
| RGB → IR  | `(512, 512)`   | large    | ~56.5 M |
| SAR → IR  | `(512, 512)`   | large    | ~56.5 M |
| SAR → RGB | `(512, 512)`   | large    | ~56.5 M |
| SAR → EO  | `(256, 256)`   | medium   | ~14.1 M |

**Observation**: GAN collapse / under-fitting observed. **Status**: Failed.

**Checkpoint**: <https://huggingface.co/BiliSakura/4th-MAVIC-T-ckpt-failed>

**Takeaways**:

- Pixel-space CUT converges faster than latent-space DDBM in this setup, but exhibited failure modes.
- Fewer parameters (~56.5 M vs ~120 M) but did not yield stable, usable results.

---

### Experiment 3: Pixel-Space DDBM

**Config**: DDBM in pixel space (no VAE). Per [roadmap § Stage 1 — Pixel-Space DDBM](roadmap.md#stage-1--pixel-space-ddbm-main). Trained all 4 tasks for 10,000 steps.

| Task      | Pixel shape    | DDBM tier | ~Params |
|-----------|----------------|-----------|---------|
| RGB → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → RGB | `(512, 512)`   | medium    | ~120 M  |
| SAR → EO  | `(256, 256)`   | small     | ~20 M   |

**Outcome**: Awesome. SAR→EO completed ~5 epochs and RGB→IR ~8 epochs—both yield appealing results. SAR→IR and SAR→RGB did not complete their first epoch (under-fitting); we will continue training on them.

**Checkpoint**: `huggingface/models/BiliSakura/4th-MAVIC-T-ckpt` (private repo; access via `HF_TOKEN` by sakura).
