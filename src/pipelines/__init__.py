"""Inference pipelines for MAVIC-T baselines."""

from .ddbm import DDBMPipeline, DDBMPipelineOutput, DDBMLatentPipeline, DDBMLatentPipelineOutput
from .bibbdm import BiBBDMPipeline, BiBBDMPipelineOutput, BiBBDMLatentPipeline, BiBBDMLatentPipelineOutput
from .ddib import DDIBPipeline, DDIBPipelineOutput, DDIBLatentPipeline, DDIBLatentPipelineOutput
from .i2sb import I2SBPipeline, I2SBPipelineOutput, I2SBLatentPipeline, I2SBLatentPipelineOutput
from .cut import CUTPipeline, CUTPipelineOutput, CUTLatentPipeline, CUTLatentPipelineOutput
from .turbo import Pix2PixTurboPipeline, CycleGANTurboPipeline, TurboPipelineOutput
