"""Production-ready self-contained pipelines for MAVIC-T models.

These pipelines can be used for inference without any external project code.
Each pipeline directory contains UNet, Scheduler, and Pipeline in one module).

Usage:
    from examples.pipelines.ddbm.pipeline import DDBMPipeline, DDBMUNet, DDBMScheduler

    unet = DDBMUNet.from_pretrained(ckpt_path, subfolder="ema_unet")
    scheduler = DDBMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
    pipeline = DDBMPipeline(unet=unet, scheduler=scheduler)
"""
