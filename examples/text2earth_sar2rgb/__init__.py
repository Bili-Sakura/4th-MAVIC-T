"""Text2Earth-based SAR2RGB training examples.

Fine-tune the pre-trained Text2Earth text2image diffusion model for SAR-to-RGB
translation via:
- ControlNet: add a ControlNet branch conditioned on SAR imagery
- InstructPix2Pix: direct image conditioning by concatenating SAR latent with
  noisy RGB latent (8-channel UNet input)
"""
