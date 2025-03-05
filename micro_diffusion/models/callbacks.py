from typing import Dict, List, Optional, Sequence
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context

import os
from PIL import Image

class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.
    
    Args:
        prompts (List[str]): List of prompts to use for evaluation.
        sampling_steps (int): Number of inference steps to use during sampling.
        guidance_scale (float): Guidance scale in classifier free guidance (scale=1 implies no classifier free guidance).
        seed (int): Random seed to use for generation. Set a seed for reproducible generation.
    """

    def __init__(self, prompts: List[str], sampling_steps: int = 30, guidance_scale: float= 5.0, seed:Optional[int]=1337):
        self.prompts = prompts
        self.sampling_steps = sampling_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    def eval_batch_end (self, state: State, logger: Logger):
        # only log once per eval epoch
        if state.eval_timestamp.get (TimeUnit.BATCH).value == 1:
            # get the model object if it has been wrapped by DDP
            model = state.model.module if isinstance(state.model, DDP) else state.model

            # Generate images
            with get_precision_context(state.precision):
                images = model.generate(
                    self.prompts,
                    num_inference_steps = self.sampling_steps,
                    guidance_scale = self.guidance_scale,
                    seed = self.seed,
                )

            save_dir = f"./generated_images/{state.run_name}/"
            os.makedirs (save_dir, exist_ok=True)

            for i, (prompt, image) in enumerate(zip(self.prompts, images)):
                image_path = os.path.join(save_dir, f"{state.timestamp.batch.value}_{i}_{prompt[:100]}.png")
                image = Image.fromarray(image) # convert tensor/array to PIL image if needed
                image.save(image_path) # save image to disk

            # Log images to tensorboard
            for prompt, image in zip (self.prompts, images):
                logger.log_images(
                    images=image,
                    name=prompt[:100],
                    step=state.timestamp.batch.value,
                    use_table=False
                )

class NaNCatcher (Callback):
    """Catches NaNs in the loss and raises an error if one is found"""

    def after_loss(self, state: State, logger:Logger):
        """Check if loss is NaN and raise an error if so."""
        if isinstance(state.loss, torch.Tensor):
            if torch.isnan(state.loss).any():
                raise RuntimeError ('Train loss contains a NaN.')
        elif isinstance(state.loss, Sequence):
            for loss in state.loss:
                if torch.isnan(loss).any():
                    raise RuntimeError('Train loss contains a NaN.')
        elif isinstance(state.loss, Dict):
            for k,v in state.loss.items():
                if torch.isnan(v).any():
                    raise RuntimeError(f'Train loss {k} contains a NaN.')
        else:
            raise TypeError (f' Loss is of type {type(state.loss)}, but should be a tensor or a list of tensors.')