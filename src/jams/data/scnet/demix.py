# Adapted from ZFTurbo/Music-Source-Separation-Training (MIT License)
# https://github.com/ZFTurbo/Music-Source-Separation-Training
# commit ccc011abf7f89dd7922bb2888d48493b575c0289, utils/model_utils.py
# Slimmed to the "generic" overlap-add inference path used by SCNet: the demucs-mode
# branch, distributed hooks, BigShifts wrapper and progress bar were removed; behaviour
# for SCNet is otherwise identical (same chunking, reflect padding, fade windowing).
"""Chunked overlap-add source separation for the vendored SCNet model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Linear fade-in/fade-out window (ones in the middle)."""
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(config, model: torch.nn.Module, mix, device: torch.device) -> dict[str, np.ndarray]:
    """Separate ``mix`` (channels, time) into per-instrument arrays.

    ``config`` needs: audio.chunk_size (or inference.chunk_size), training.instruments,
    inference.num_overlap, inference.batch_size, training.use_amp.
    """
    mix = torch.tensor(np.asarray(mix), dtype=torch.float32)

    chunk_size = getattr(config.inference, "chunk_size", None) or config.audio.chunk_size
    instruments = list(config.training.instruments)
    num_instruments = len(instruments)
    num_overlap = config.inference.num_overlap

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = _get_windowing_array(chunk_size, fade_size)
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size
    use_amp = bool(getattr(config.training, "use_amp", True)) and device.type == "cuda"

    with torch.autocast(device_type="cuda", enabled=use_amp), torch.inference_mode():
        req_shape = (num_instruments,) + tuple(mix.shape)
        result = torch.zeros(req_shape, dtype=torch.float32)
        counter = torch.zeros(req_shape, dtype=torch.float32)

        i = 0
        batch_data: list[torch.Tensor] = []
        batch_locations: list[tuple[int, int]] = []
        while i < mix.shape[1]:
            part = mix[:, i:i + chunk_size].to(device)
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = nn.functional.pad(
                part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
            )
            batch_data.append(part)
            batch_locations.append((i, chunk_len))
            i += step

            if len(batch_data) >= batch_size or i >= mix.shape[1]:
                arr = torch.stack(batch_data, dim=0)
                x = model(arr)

                window = windowing_array.clone()
                if i - step == 0:  # first chunk: no fade-in
                    window[:fade_size] = 1
                elif i >= mix.shape[1]:  # last chunk: no fade-out
                    window[-fade_size:] = 1

                for j, (start, seg_len) in enumerate(batch_locations):
                    result[..., start:start + seg_len] += (
                        x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                    )
                    counter[..., start:start + seg_len] += window[..., :seg_len]

                batch_data.clear()
                batch_locations.clear()

        estimated_sources = (result / counter).cpu().numpy()
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)

        if length_init > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

    return dict(zip(instruments, estimated_sources, strict=True))
