# Denoising Diffusion Bridge Models (ICLR 2024)

Official Implementation of [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948).

<p align="center">
  <img src="assets/teaser.png" width="100%"/>
</p>


# Installation

## Quick Install (Inference Only)

For using DDBM with the diffusers API for inference:

```sh
pip install torch torchvision
pip install -e .
```

## Full Install (Training + Inference)

To install all packages for training and evaluation:

```sh
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
conda install -c conda-forge mpi4py openmpi
pip install -e ".[full]"
```

# 🤗 Hugging Face Diffusers Integration

This codebase is built on the [Hugging Face diffusers](https://github.com/huggingface/diffusers) library for easy use and integration!

## Quick Start

```python
import torch
from ddbm import DDBMScheduler, DDBMPipeline

# 1. Create scheduler
scheduler = DDBMScheduler(
    sigma_max=1.0,          # Maximum sigma (use 1.0 for VP mode)
    sigma_min=0.0001,       # Minimum sigma
    pred_mode='vp',         # 'vp' or 've' schedule
    beta_d=2.0,             # Beta_d parameter
    beta_min=0.1,           # Beta_min parameter
)

# 2. Load your trained model
from ddbm import create_model, model_and_diffusion_defaults
model = create_model(
    image_size=64,
    in_channels=3,
    num_channels=128,
    num_res_blocks=2,
    unet_type='adm',
    condition_mode='concat',
    **{k: v for k, v in model_and_diffusion_defaults().items()
       if k not in ['image_size', 'in_channels', 'num_channels', 'num_res_blocks']}
)
model.load_state_dict(torch.load("your_checkpoint.pt"))
model.eval()

# 3. Create pipeline
pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

# 4. Run inference
result = pipeline(
    source_image=your_source_image,  # Can be tensor or PIL Image
    num_inference_steps=40,
    guidance=1.0,
    churn_step_ratio=0.33,
    output_type='pil',
)

# 5. Access results (using diffusers-style output)
images = result.images  # List of PIL Images
nfe = result.nfe        # Number of function evaluations
```

## Scheduler-Only Usage

For custom sampling loops, use the scheduler directly:

```python
from ddbm import DDBMScheduler

scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
scheduler.set_timesteps(40)

# Add noise for training (bridge process)
noisy = scheduler.add_noise(clean_samples, noise, timesteps, target_samples)

# Access sigmas for sampling
print(f"Sigmas: {scheduler.sigmas}")
print(f"Init noise sigma: {scheduler.init_noise_sigma}")

# Save and load scheduler config
scheduler.save_config("./my_scheduler")
loaded_scheduler = DDBMScheduler.from_config("./my_scheduler")
```

## Diffusers-Style Training

Train DDBM models using the accelerate library for distributed training:

```sh
# Single GPU training
python scripts/train_ddbm_diffusers.py \
    --dataset_name edges2handbags \
    --data_dir /path/to/data \
    --output_dir ./outputs/ddbm-e2h \
    --resolution 64 \
    --train_batch_size 32 \
    --num_epochs 100 \
    --pred_mode vp

# Multi-GPU training with accelerate
accelerate launch scripts/train_ddbm_diffusers.py \
    --dataset_name edges2handbags \
    --data_dir /path/to/data \
    --output_dir ./outputs/ddbm-e2h \
    --mixed_precision fp16 \
    --use_ema
```

See [examples/diffusers_example.py](examples/diffusers_example.py) for more detailed examples.

---

# Pre-trained models

We provide pretrained checkpoints via Huggingface repo [here](https://huggingface.co/alexzhou907/DDBM). It includes models trained on two image-to-image datasets using Variance-Preserving (VP) schedules:

 * DDBM on Edges2Handbags (VP): [ddbm_e2h_vp_ema.pt](https://huggingface.co/alexzhou907/DDBM/resolve/main/e2h_ema_0.9999_420000.pt)
 * DDBM on DIODE (VP): [ddbm_diode_vp_ema.pt](https://huggingface.co/alexzhou907/DDBM/resolve/main/diode_ema_0.9999_440000.pt)

# Datasets

For Edges2Handbags, please follow instructions from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md).
For DIODE, please download appropriate datasets from [here](https://diode-dataset.org/).


# Model training and sampling

We provide bash files [train_ddbm.sh](train_ddbm.sh) and [sample_ddbm.sh](sample_ddbm.sh) for model training and sampling. 

Simply set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` specifies which dataset to use. We only support `e2h` for Edges2Handbags and `diode` for DIODE. For each dataset, make sure to set the respective `DATA_DIR` variable in `args.sh` to your dataset path.
- `SCHEDULE_TYPE` denotes the noise schedule type. Only `ve` and `vp` are recommended. `ve_simple` and `vp_simple` are their naive baselines.

To train, run
```
bash train_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE 

# to resume, set CKPT to your checkpoint, or it will automatically resume from your last checkpoint based on your experiment name.

bash train_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE $CKPT
```


For inference, additional variables need to be set:
- `MODEL_PATH` is your checkpoint to be evaluated.
- `CHURN_STEP_RATIO` is the ratio of step that's used for stochastic Euler step (see paper for details). Default recommendation is `0.33`. Lower value generally degrades performance. For better value setting please refer to the paper.
- `GUIDANCE` is the `w` parameter specified in the paper. Default recommendation is `1` for VP schedules and anything less than `1` produces significantly worse results. However, for VE schedules, this value (ranging from `0` to `1`) does not affect generation too much. . For better value setting please refer to the paper.
- `SPLIT` denotes which split you use for testing. Only `train` and `test` are supported.
To sample, run
```
bash sample_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE $MODEL_PATH $CHURN_STEP_RATIO $GUIDANCE $SPLIT
```
This script will aggregate all samples into `.npz` file into your experiment folder ready for quantitative evaluation.


# Evaluations

One can evaluate samples with [evaluations/evaluator.py](evaluations/evaluator.py). We also provide the reference statistics in our Huggingface [repo](https://huggingface.co/alexzhou907/DDBM):
- Reference stats for Edge2Handbags: [e2h_ref_stats.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/edges2handbags_ref_64_data.npz).
- Reference stats for DIODE: [diode_ref_stats.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/diode_ref_256_data.npz).

To evaluate, set `REF_PATH` to path of your reference stats and `SAMPLE_PATH` to your generated `.npz` path. You can additionally specify the metrics to use via `--metric`. We only support `fid` and `lpips`.
```
python $REF_PATH $SAMPLE_PATH --metric $YOUR_METRIC
```

# Toubleshoot

We noticed that on some machines `mpiexec` errors out with
```
--------------------------------------------------------------------------
MPI_INIT has failed because at least one MPI process is unreachable
from another.  This *usually* means that an underlying communication
plugin -- such as a BTL or an MTL -- has either not loaded or not
allowed itself to be used.  Your MPI job will now abort.

You may wish to try to narrow down the problem;  

 * Check the output of ompi_info to see which BTL/MTL plugins are
   available.
 * Run your application with MPI_THREAD_SINGLE.  
 * Set the MCA parameter btl_base_verbose to 100 (or mtl_base_verbose,
   if using MTL-based communications) to see exactly which
   communication plugins were considered and/or discarded.
--------------------------------------------------------------------------
```

In this case, you can try adding `--mca btl vader,self` to `mpiexec` command before `python` run.

During evaluation, if you see significantly high LPIPS or MSE scores, this is likely due to mismatch in order between your generation and the reference stats. This may be due to the multiprocess gathering of results returning the incorrect order. Please make sure the order is correct for your generation, or regenerate the reference stats by yourself.


# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{zhou2023denoising,
  title={Denoising diffusion bridge models},
  author={Zhou, Linqi and Lou, Aaron and Khanna, Samar and Ermon, Stefano},
  journal={arXiv preprint arXiv:2309.16948},
  year={2023}
}
```
