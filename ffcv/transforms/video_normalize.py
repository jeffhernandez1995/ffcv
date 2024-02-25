"""
Video normalization
"""

from collections.abc import Sequence
from typing import Tuple

import numpy as np
import torch as ch
from numpy import dtype
from numpy.random import rand
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler


def ch_dtype_from_numpy(dtype):
    return ch.from_numpy(np.zeros((), dtype=dtype)).dtype


class NormalizeVideo(Operation):
    def __init__(self, mean: np.ndarray, std: np.ndarray, type: np.dtype):
        super().__init__()
        table = (np.arange(256)[:, None] - mean[None, :]) / std[None, :]
        self.original_dtype = type
        table = table.astype(type)
        if type == np.float16:
            type = np.int16
        self.dtype = type
        table = table.view(type)
        self.lookup_table = table
        self.mode = 'cpu'

    def generate_code(self) -> Callable:
        if self.mode == 'cpu':
            return self.generate_code_cpu()
        return self.generate_code_gpu()

    def generate_code_gpu(self) -> Callable:
        import cupy as cp
        import pytorch_pfn_extras as ppe

        tn = np.zeros((), dtype=self.dtype).dtype.name
        kernel = cp.ElementwiseKernel(
            f'uint8 input, raw {tn} table', f'{tn} output', 'output = table[input * 3 + i % 3];')
        final_type = ch_dtype_from_numpy(self.original_dtype)

        def normalize_convert(videos, result):
            B, T, C, H, W = videos.shape
            table = self.lookup_table.view(-1)
            # assert videos.is_contiguous(memory_format=ch.channels_last), 'Videos need to be in channel last'
            result = result[:B * T]
            result_c = result.view(-1)
            videos = videos.permute(0, 1, 3, 4, 2).reshape(-1)
            print(videos.device)

            current_stream = ch.cuda.current_stream()
            with ppe.cuda.stream(current_stream):
                kernel(videos, table, result_c)

            final_result = result.reshape(B, T, H, W, C).permute(0, 1, 3, 4, 2)
            # assert final_result.is_contiguous(memory_format=ch.channels_last), 'Videos need to be in channel last'

            return final_result.view(final_type)

        return normalize_convert

    def generate_code_cpu(self) -> Callable:
        table = self.lookup_table.view(dtype=self.dtype)
        my_range = Compiler.get_iterator()

        def normalize_convert(videos, result, indices):
            result_flat = result.reshape(result.shape[0], result.shape[1], -1, 3)
            num_pixels = result_flat.shape[2]
            for i in my_range(len(indices)):
                for t in range(videos.shape[1]):  # Iterate over the time dimension
                    video_frame = videos[i, t].reshape(num_pixels, 3)
                    for px in range(num_pixels):
                        for c in range(3):  # Iterate over channels
                            result_flat[i, t, px, c] = table[video_frame[px, c], c]

            return result

        normalize_convert.is_parallel = True
        normalize_convert.with_indices = True
        return normalize_convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:

        if previous_state.device == ch.device('cpu'):
            new_state = replace(previous_state, jit_mode=True, dtype=self.dtype)
            return new_state, AllocationQuery(
                shape=previous_state.shape,
                dtype=self.dtype,
                device=previous_state.device
            )

        else:
            self.mode = 'gpu'
            new_state = replace(previous_state, dtype=self.dtype)

            gpu_type = ch_dtype_from_numpy(self.dtype)


            # Copy the lookup table into the proper device
            try:
                self.lookup_table = ch.from_numpy(self.lookup_table)
            except TypeError:
                pass  # This is alredy a tensor
            self.lookup_table = self.lookup_table.to(previous_state.device)

            return new_state, AllocationQuery(
                shape=previous_state.shape,
                device=previous_state.device,
                dtype=gpu_type
            )
