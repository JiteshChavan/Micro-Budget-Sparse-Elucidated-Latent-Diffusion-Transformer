import os
import time
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

from micro_diffusion.datasets.prepare.textcaps.base import (
    build_streaming_textcaps_precompute_dataloader
)