from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset

from micro_diffusion.models.utils import UniversalTokenizer


class StreamingTextcapsDatasetForPreCompute(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
            self,
            streams: Sequence[Stream],
            transforms_list: List[Callable],
            batch_size: int,
            tokenizer_name: str,
            shuffle: bool = False,
            caption_key: str = 'caption_syn_pixart_llava15'
    ):
        super().__init__(streams=streams, shuffle=shuffle, batch_size=batch_size)