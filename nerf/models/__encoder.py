from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder:
    def __init__(
        self, input_dims: int, max_freq: int, num_freqs: int, log_sampling: bool
    ) -> None:
        self.encode_fns: list[Callable] = []
        self.out_dim = input_dims

        self.encode_fns.append(lambda x: x)

        if log_sampling:
            freq_bands: torch.Tensor = 2.0 ** torch.linspace(
                0.0, max_freq, steps=num_freqs
            )
        else:
            freq_bands: torch.Tensor = torch.linspace(
                2.0**0.0, 2.0**max_freq, steps=num_freqs
            )

        for freq in freq_bands:
            for fn in [torch.sin, torch.cos]:
                self.encode_fns.append(lambda x, fn=fn, freq=freq: fn(x * freq))
                self.out_dim += input_dims

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.encode_fns], dim=-1)
