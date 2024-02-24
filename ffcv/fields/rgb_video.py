from abc import ABCMeta, abstractmethod
import ffmpeg
from dataclasses import replace
from typing import Callable, Tuple, Type, Optional

import cv2
import numpy as np


from ffcv.fields.base import Field, ARG_TYPE
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.libffcv import imdecode, memcpy, resize_crop


def encode_jpeg(numpy_image, quality):
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    success, result = cv2.imencode('.jpg', numpy_image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    if not success:
        raise ValueError("Impossible to encode image in jpeg")

    return result.reshape(-1)


def get_random_crop(height, width, scale, ratio):
    area = float(height) * float(width)
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def get_center_crop(height, width, _, ratio):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c


def resizer(frames, target_resolution):
    # frames [T, H, W, C]
    if target_resolution is None:
        return frames
    original_size = np.array([frames.shape[2], frames.shape[1]])
    ratio = target_resolution / original_size.min()
    if ratio < 1:
        new_size = (ratio * original_size).astype(int)
        frames = np.array([
            cv2.resize(frame, tuple(new_size), interpolation=cv2.INTER_AREA)
            for frame in frames
        ])
    return frames


class SimpleRGBVideoDecoder(Operation):
    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']

        max_height = heights.max()
        max_width = widths.max()
        max_num_frames = self.metadata['num_frames'].max()
        min_num_frames = self.metadata['num_frames'].min()
        min_height = heights.min()
        min_width = widths.min()

        if min_height != max_height or min_width != max_width:
            msg = "We only support videos with the same resolution"
            raise TypeError(msg)
        if min_num_frames != max_num_frames:
            msg = "We only support videos with the same number of frames"
            raise TypeError(msg)

        video_size = np.uint64(self.metadata['size'].max())
        jpeg_encoded_video_shape = (video_size,)

        itemsize = np.dtype('<u8').itemsize
        splits_shape = (np.uint64(itemsize * max_num_frames),)

        temp_frame_shape = (max_height * max_width * np.uint64(3),)
        final_shape = (
            max_num_frames,
            np.uint64(self.output_size),
            np.uint64(self.output_size),
            np.uint64(3)
        )

        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=temp_frame_shape, dtype=my_dtype),
            (
                AllocationQuery(final_shape, my_dtype),
                AllocationQuery(splits_shape, my_dtype),
                AllocationQuery(jpeg_encoded_video_shape, my_dtype),
                AllocationQuery(temp_frame_shape, my_dtype)
            )
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        imdecode_c = Compiler.compile(imdecode)

        my_range = Compiler.get_iterator()
        my_memcpy = Compiler.compile(memcpy)

        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, idx_storage, jpeg_video_storage, frame_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                video_data = mem_read(field['data_ptr'], storage_state)
                indices_data = mem_read(field['splits'], storage_state)

                height, width, num_frames = field['height'], field['width'], field['num_frames']

                my_memcpy(video_data, jpeg_video_storage[dst_ix])
                my_memcpy(indices_data, idx_storage[dst_ix])

                temp_splits = idx_storage[dst_ix].view(np.uint64)
                temp_video = jpeg_video_storage[dst_ix]

                temp_video = np.split(temp_video, temp_splits)[:-1]
                for idx in range(len(temp_video)):
                    temp_buffer = frame_storage[dst_ix]
                    imdecode_c(
                        temp_video[idx],
                        temp_buffer,
                        height, width, height, width, 0, 0, 1, 1, False, False
                    )
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)
                    destination[dst_ix, idx, ...] = temp_buffer
            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


class ResizedCropRGBVideoDecoder(SimpleRGBVideoDecoder, metaclass=ABCMeta):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        max_num_frames = np.uint64(self.metadata['num_frames'].max())
        min_num_frames = np.uint64(self.metadata['num_frames'].min())
        if min_num_frames != max_num_frames:
            msg = "We only support videos with the same number of frames"
            raise TypeError(msg)
        max_width = np.uint64(widths.max())
        max_height = np.uint64(heights.max())

        video_size = np.uint64(self.metadata['size'].max())
        jpeg_encoded_video_shape = (video_size,)

        itemsize = np.dtype('<u8').itemsize
        splits_shape = (np.uint64(itemsize * max_num_frames),)

        temp_frame_shape = (max_height * max_width * np.uint64(3),)
        final_shape = (
            max_num_frames,
            np.uint64(self.output_size),
            np.uint64(self.output_size),
            np.uint64(3)
        )

        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=final_shape, dtype=my_dtype),
            (
                AllocationQuery(final_shape, my_dtype),
                AllocationQuery(splits_shape, my_dtype),
                AllocationQuery(jpeg_encoded_video_shape, my_dtype),
                AllocationQuery(temp_frame_shape, my_dtype)
            )
        )


    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        imdecode_c = Compiler.compile(imdecode)

        my_range = Compiler.get_iterator()
        my_memcpy = Compiler.compile(memcpy)

        resize_crop_c = Compiler.compile(resize_crop)
        get_crop_c = Compiler.compile(self.get_crop_generator)

        scale = self.scale
        ratio = self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, idx_storage, jpeg_video_storage, frame_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                video_data = mem_read(field['data_ptr'], storage_state)
                indices_data = mem_read(field['splits'], storage_state)

                height, width, num_frames = field['height'], field['width'], field['num_frames']

                my_memcpy(video_data, jpeg_video_storage[dst_ix])
                my_memcpy(indices_data, idx_storage[dst_ix])

                temp_splits = idx_storage[dst_ix].view(np.uint64)
                temp_video = jpeg_video_storage[dst_ix]

                temp_video = np.split(temp_video, temp_splits)[:-1]
                i, j, h, w = get_crop_c(height, width, scale, ratio)
                for idx in range(len(temp_video)):
                    temp_buffer = frame_storage[dst_ix]
                    imdecode_c(
                        temp_video[idx],
                        temp_buffer,
                        height, width, height, width, 0, 0, 1, 1, False, False
                    )
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)

                    resize_crop_c(
                        temp_buffer,
                        i, i + h, j, j + w,
                        destination[dst_ix, idx, ...]
                    )

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


class RandomResizedCropRGBVideoDecoder(ResizedCropRGBVideoDecoder):
    def __init__(self, output_size, scale=(0.08, 1.0), ratio=(0.75, 4/3)):
        super().__init__(output_size)
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size

    @property
    def get_crop_generator(self):
        return get_random_crop


class CenterCropRGBVideoDecoder(ResizedCropRGBVideoDecoder):
    # output size: resize crop size -> output size
    def __init__(self, output_size, ratio):
        super().__init__(output_size)
        self.scale = None
        self.ratio = ratio

    @property
    def get_crop_generator(self):
        return get_center_crop


class RGBVideoField(Field):
    def __init__(
        self,
        min_resolution: int = None,
        quality: int = 90,
    ) -> None:
        self.min_resolution = min_resolution
        self.quality = int(quality)

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('width', '<u2'),
            ('num_frames', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
            ('splits', '<u8'),
            ('size', '<u8'),
        ])

    def get_decoder_class(self) -> Type[Operation]:
        return SimpleRGBVideoDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return RGBVideoField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, video, malloc):
        if not isinstance(video, np.ndarray):
            raise TypeError(f"Unsupported video type {type(video)}")
        
        if video.ndim != 4:
            raise ValueError(f"Unsupported video shape {video.shape}")

        video = resizer(video, self.min_resolution)

        num_frames, height, width, channels = video.shape

        video = [encode_jpeg(frame, self.quality) for frame in video]

        sizes = np.cumsum([len(x) for x in video], dtype=np.uint64)
        video = np.concatenate(
            video,
            axis=0,
            dtype=np.uint8
        )
        destination['width'] = width
        destination['height'] = height
        destination['num_frames'] = num_frames
        destination['data_ptr'], storage = malloc(video.nbytes)
        storage[:] = video
        destination['size'] = video.size
        destination['splits'], storage = malloc(sizes.nbytes)
        storage[:] = sizes.reshape(-1).view('<u1')
