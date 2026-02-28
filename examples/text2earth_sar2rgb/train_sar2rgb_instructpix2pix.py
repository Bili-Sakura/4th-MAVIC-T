#!/usr/bin/env python
"""Train SAR2RGB via InstructPix2Pix-style conditioning on Text2Earth.

Fine-tune the Text2Earth UNet by adding direct image conditioning: concatenate
SAR latent with noisy RGB latent as 8-channel input (InstructPix2Pix style).
Only the UNet is trained; VAE and text encoder are frozen.

Based on: libs/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder

from examples.text2earth_sar2rgb.dataset_utils import (
    MavicTSAR2RGBDataset,
    load_paired_manifest,
    load_paired_manifest,
    load_sar2rgb_train_records,
)
from src.utils.training_utils import (
    create_optimizer,
    lambda_repa_cosine,
    log_validation_images_to_trackers,
)

logger = get_logger(__name__)

DEFAULT_TEXT2EARTH_PATH = "models/lcybuaa/Text2Earth"


def parse_args():
    parser = argparse.ArgumentParser(description="Train InstructPix2Pix-style SAR2RGB on Text2Earth.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=DEFAULT_TEXT2EARTH_PATH,
        help="Path to Text2Earth or compatible SD model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpt/text2earth_sar2rgb_instructpix2pix",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default="models/lcybuaa/Text2Earth/vae",
        help="VAE model to use. Default: models/lcybuaa/Text2Earth/vae.",
    )
    parser.add_argument(
        "--refined_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sar2rgb_sup_manifest",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt",
    )
    parser.add_argument(
        "--paired_val_manifest",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt",
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt",
    )
    parser.add_argument(
        "--use_augmented",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="18_GOOGLE_LEVEL_ a satellite optical image",
        help="Text prompt for conditioning. Use N_GOOGLE_LEVEL_ prefix for resolution: N in [10,18] maps to 2^(17-N)m (e.g. 17→1m, 18→0.5m).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "prodigy"],
        help="Optimizer: adamw or prodigy.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--prodigy_d0",
        type=float,
        default=1e-5,
        help="Prodigy d0 parameter (used when optimizer_type=prodigy).",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.05,
        help="Dropout for SAR conditioning (CFG at inference).",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=0,
        help="Run validation every N steps. 0 disables validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Logging backend: tensorboard, swanlab, wandb, etc.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub repo id for push_to_hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="HuggingFace token for push_to_hub.",
    )
    parser.add_argument(
        "--swanlab_experiment_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--swanlab_description",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--swanlab_tags",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--swanlab_init_kwargs_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Text2Earth resolution class label.",
    )
    # Representation alignment (REPA)
    parser.add_argument(
        "--use_rep_alignment",
        action="store_true",
        default=False,
        help="Enable REPA representation alignment loss (MaRS-RGB on target).",
    )
    parser.add_argument(
        "--rep_alignment_model_path",
        type=str,
        default="./models/BiliSakura/MaRS-Base-RGB",
        help="Path to REPA teacher encoder (MaRS-Base-RGB for SAR2RGB).",
    )
    parser.add_argument(
        "--lambda_rep_alignment",
        type=float,
        default=1.0,
        help="Weight for REPA alignment loss.",
    )
    parser.add_argument(
        "--lambda_rep_alignment_decay_steps",
        type=int,
        default=0,
        help="Steps over which to cosine-decay REPA lambda (0 = no decay).",
    )
    parser.add_argument(
        "--lambda_rep_alignment_end",
        type=float,
        default=0.0,
        help="End value for REPA lambda decay.",
    )
    return parser.parse_args()


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    sar_pixel_values = torch.stack([e["sar_pixel_values"] for e in examples])
    sar_pixel_values = sar_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([e["input_ids"] for e in examples])
    return {
        "pixel_values": pixel_values,
        "sar_pixel_values": sar_pixel_values,
        "input_ids": input_ids,
    }


@torch.no_grad()
def log_validation_samples(
    args,
    accelerator: Accelerator,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    val_dataloader,
    weight_dtype,
    global_step: int,
    has_class_embedding: bool,
    class_label: int,
):
    if val_dataloader is None or args.validation_num_images <= 0:
        return

    logger.info("Running sample inference validation at step %d ...", global_step)

    unwrapped_unet = accelerator.unwrap_model(unet)
    was_training = unwrapped_unet.training
    unwrapped_unet.eval()

    val_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
    val_scheduler.set_timesteps(args.validation_num_inference_steps, device=accelerator.device)

    sample_dir = Path(args.output_dir) / "test_results" / f"step-{global_step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    logged_grids = []
    saved = 0
    for batch in val_dataloader:
        remaining = args.validation_num_images - saved
        if remaining <= 0:
            break
        take_n = min(remaining, batch["sar_pixel_values"].shape[0])

        sar_pixel_values = batch["sar_pixel_values"][:take_n].to(accelerator.device, dtype=weight_dtype)
        target_pixel_values = batch["pixel_values"][:take_n].to(accelerator.device, dtype=weight_dtype)
        input_ids = batch["input_ids"][:take_n].to(accelerator.device)

        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
        sar_rgb = sar_pixel_values.repeat(1, 3, 1, 1)
        sar_latents = vae.encode(sar_rgb).latent_dist.sample()
        sar_latents = sar_latents * vae.config.scaling_factor
        latents = torch.randn_like(sar_latents)

        for t in val_scheduler.timesteps:
            latent_input = torch.cat([latents, sar_latents], dim=1)
            unet_kwargs = dict(return_dict=False)
            if has_class_embedding:
                unet_kwargs["class_labels"] = torch.full(
                    (take_n,),
                    class_label,
                    dtype=torch.long,
                    device=latents.device,
                )
            with accelerator.autocast():
                noise_pred = unwrapped_unet(
                    latent_input,
                    t,
                    encoder_hidden_states,
                    **unet_kwargs,
                )[0]
            latents = val_scheduler.step(noise_pred, t, latents).prev_sample

        pred_pixels = vae.decode(latents / vae.config.scaling_factor).sample
        pred_pixels = ((pred_pixels / 2) + 0.5).clamp(0, 1)
        src_vis = ((sar_pixel_values + 1) / 2).repeat(1, 3, 1, 1).clamp(0, 1)
        tgt_vis = ((target_pixel_values + 1) / 2).clamp(0, 1)

        for i in range(take_n):
            src_img = Image.fromarray(
                (src_vis[i].detach().float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            )
            pred_img = Image.fromarray(
                (pred_pixels[i].detach().float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            )
            tgt_img = Image.fromarray(
                (tgt_vis[i].detach().float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            )
            grid = make_image_grid([src_img, pred_img, tgt_img], rows=1, cols=3)
            grid.save(sample_dir / f"sample_{saved:03d}.png")
            logged_grids.append(grid)
            saved += 1
            if saved >= args.validation_num_images:
                break

    logger.info("Saved %d sample inference grids to %s", saved, sample_dir)
    if logged_grids:
        log_validation_images_to_trackers(
            accelerator,
            logged_grids,
            global_step,
            tag="validation/samples",
        )
    if was_training:
        unwrapped_unet.train()


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    # InstructPix2Pix: 8 channels = 4 (noisy latent) + 4 (SAR latent)
    # SAR is 1ch; we replicate to 3ch for VAE encode, then get 4ch latent
    logger.info("Initializing InstructPix2Pix-style 8-channel UNet from Text2Earth UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerate.__version__ >= "0.16.0":
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    if weights:
                        weights.pop()
                if rep_alignment_module is not None and rep_alignment_module.projector is not None:
                    torch.save(
                        rep_alignment_module.projector.state_dict(),
                        os.path.join(output_dir, "rep_projector.pt"),
                    )

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            while len(models) > 0:
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
            if rep_alignment_module is not None and rep_alignment_module.projector is not None:
                projector_path = os.path.join(input_dir, "rep_projector.pt")
                if os.path.isfile(projector_path):
                    rep_alignment_module.projector.load_state_dict(
                        torch.load(projector_path, map_location="cpu"),
                        strict=True,
                    )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Representation alignment (REPA) — MaRS-RGB encodes target RGB
    rep_alignment_module = None
    if args.use_rep_alignment and args.rep_alignment_model_path:
        from src.utils.rep_alignment import MaRSRGBAlignment
        rep_alignment_module = MaRSRGBAlignment(args.rep_alignment_model_path)
        rep_alignment_module.build_projector(3)  # target_channels for RGB
        logger.info(
            "REPA enabled (model=%s, lambda=%s%s)",
            args.rep_alignment_model_path,
            args.lambda_rep_alignment,
            f"→{args.lambda_rep_alignment_end} cos decay over {args.lambda_rep_alignment_decay_steps} steps"
            if args.lambda_rep_alignment_decay_steps > 0 else "",
        )

    train_params = list(unet.parameters())
    if rep_alignment_module is not None and rep_alignment_module.projector is not None:
        train_params += list(rep_alignment_module.projector.parameters())

    if args.optimizer_type.lower() == "prodigy":
        optimizer = create_optimizer(
            train_params,
            optimizer_type="prodigy",
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            prodigy_d0=args.prodigy_d0,
        )
    elif args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                train_params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam")
    else:
        optimizer = torch.optim.AdamW(
            train_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset
    records = load_sar2rgb_train_records(
        refined_root=args.refined_root,
        sar2rgb_sup_manifest=args.sar2rgb_sup_manifest,
        exclude_file=args.exclude_file,
        paired_val_manifest=args.paired_val_manifest,
        use_augmented=args.use_augmented,
    )
    train_dataset = MavicTSAR2RGBDataset(
        records=records,
        resolution=args.resolution,
        caption=args.caption,
        use_random_crop=True,
        use_horizontal_flip=True,
        use_vertical_flip=False,
        tokenizer=tokenizer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = None
    if args.validation_steps > 0 and args.paired_val_manifest:
        try:
            val_records = load_paired_manifest(args.paired_val_manifest)
            if val_records:
                val_dataset = MavicTSAR2RGBDataset(
                    records=val_records,
                    resolution=args.resolution,
                    caption=args.caption,
                    use_random_crop=False,
                    use_horizontal_flip=False,
                    use_vertical_flip=False,
                    tokenizer=tokenizer,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    collate_fn=collate_fn,
                    batch_size=args.train_batch_size,
                    num_workers=0,
                )
                logger.info(f"Validation dataset: {len(val_records)} pairs from {args.paired_val_manifest}")
        except FileNotFoundError:
            logger.warning(f"Validation manifest not found: {args.paired_val_manifest}, skipping validation")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_unet.to(accelerator.device)
    if rep_alignment_module is not None:
        rep_alignment_module = rep_alignment_module.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    has_class_embedding = getattr(unwrap_model(unet), "class_embedding", None) is not None
    class_label = args.class_label

    # Null conditioning for CFG
    with torch.no_grad():
        null_ids = tokenizer(
            [""],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        null_conditioning = text_encoder(null_ids, return_dict=False)[0]

    if accelerator.is_main_process:
        tracker_config = {k: str(v) for k, v in vars(args).items()}
        init_kwargs = None
        if args.report_to and "swanlab" in args.report_to.lower():
            swanlab_kwargs = {"experiment_name": args.swanlab_experiment_name or "text2earth_sar2rgb_instructpix2pix"}
            if args.swanlab_description:
                swanlab_kwargs["description"] = args.swanlab_description
            if args.swanlab_tags:
                swanlab_kwargs["tags"] = [t.strip() for t in args.swanlab_tags.split(",") if t.strip()]
            if args.swanlab_init_kwargs_json:
                import json
                swanlab_kwargs.update(json.loads(args.swanlab_init_kwargs_json))
            init_kwargs = {"swanlab": swanlab_kwargs}
        accelerator.init_trackers(
            "text2earth_sar2rgb_instructpix2pix",
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    for epoch in range(first_epoch, math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Target: RGB latents
                edited_pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(edited_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SAR conditioning: 1ch -> 3ch (replicate) -> VAE encode -> 4ch latent
                sar_rgb = batch["sar_pixel_values"].repeat(1, 3, 1, 1)  # (B, 3, H, W) in [-1, 1]
                sar_rgb = sar_rgb.to(dtype=weight_dtype)
                sar_latents = vae.encode(sar_rgb).latent_dist.sample()
                sar_latents = sar_latents * vae.config.scaling_factor

                # Conditioning dropout
                if args.conditioning_dropout_prob is not None and args.conditioning_dropout_prob > 0:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    image_mask = (random_p >= args.conditioning_dropout_prob).to(sar_latents.dtype)
                    image_mask = image_mask.view(bsz, 1, 1, 1)
                    sar_latents = image_mask * sar_latents

                concatenated_noisy_latents = torch.cat([noisy_latents, sar_latents], dim=1)  # (B, 8, H, W)

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device), return_dict=False
                )[0]
                if args.conditioning_dropout_prob is not None and args.conditioning_dropout_prob > 0:
                    prompt_mask = (random_p < 2 * args.conditioning_dropout_prob).view(bsz, 1, 1)
                    encoder_hidden_states = torch.where(
                        prompt_mask,
                        null_conditioning.expand(bsz, -1, -1),
                        encoder_hidden_states,
                    )

                unet_kwargs = dict(return_dict=False)
                if has_class_embedding:
                    class_labels = torch.full(
                        (bsz,), class_label, dtype=torch.long, device=latents.device
                    )
                    unet_kwargs["class_labels"] = class_labels

                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    **unet_kwargs,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                rep_loss = None

                # Optional REPA: align predicted RGB with MaRS-RGB teacher
                if rep_alignment_module is not None:
                    alpha_prod_t = noise_scheduler.alphas_cumprod.to(
                        device=timesteps.device, dtype=latents.dtype
                    )[timesteps]
                    dims = len(latents.shape) - 1
                    alpha_prod_t = alpha_prod_t.view(-1, *([1] * dims))
                    beta_prod_t = 1 - alpha_prod_t
                    pred_x0 = (
                        noisy_latents - beta_prod_t**0.5 * model_pred
                    ) / alpha_prod_t**0.5
                    pred_x0 = pred_x0 / vae.config.scaling_factor
                    pred_pixels = vae.decode(pred_x0.to(weight_dtype)).sample
                    with torch.no_grad():
                        enc_feats = rep_alignment_module.extract_features(
                            edited_pixel_values.to(accelerator.device)
                        )
                    rep_loss = rep_alignment_module.compute_alignment_loss(
                        pred_pixels, enc_feats
                    )
                    lambda_repa = (
                        lambda_repa_cosine(
                            global_step,
                            args.lambda_rep_alignment,
                            args.lambda_rep_alignment_end,
                            args.lambda_rep_alignment_decay_steps,
                        )
                        if args.lambda_rep_alignment_decay_steps > 0
                        else args.lambda_rep_alignment
                    )
                    loss = loss + lambda_repa * (rep_loss + 1.0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(train_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for d in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, d))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if rep_loss is not None:
                logs["loss/repa"] = rep_loss.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        unet = unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        if rep_alignment_module is not None and rep_alignment_module.projector is not None:
            torch.save(
                rep_alignment_module.projector.state_dict(),
                os.path.join(args.output_dir, "rep_projector.pt"),
            )
        logger.info(f"Saved final UNet to {args.output_dir}")

        if args.push_to_hub:
            hub_token = args.hub_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
            repo_id = args.hub_model_id or Path(args.output_dir).name
            # Always use official endpoint for write operations.
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
            create_repo(repo_id=repo_id, exist_ok=True, token=hub_token)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["checkpoint-*"],
                token=hub_token,
            )
            logger.info(f"Pushed to Hub: {repo_id}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
