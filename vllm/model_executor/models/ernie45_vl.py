# coding=utf-8

"""
Ernie-45T VL Model
"""
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor
from transformers import BatchFeature
from transformers import PretrainedConfig
# from transformers.models.ernie4_5_moe_vl import Ernie_45T_VLProcessor
from vllm.transformers_utils.processors.ernie45_vl import Ernie_45T_VLProcessor, Ernie_45T_VLImageProcessor, smart_resize


from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.forward_context import get_forward_context
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsLoRA
from vllm.model_executor.models import SupportsMultiModal
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.ernie45_vit import Ernie4_5_VisionTransformer
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)

from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PlaceholderFeaturesInfo, PromptUpdate)

from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)


from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend


logger = init_logger(__name__)


# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class Ernie4_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Ernie4_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - list[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Ernie4_5_VLImageInputs = Union[Ernie4_5_VLImagePixelInputs,
                           Ernie4_5_VLImageEmbeddingInputs]

class Ernie4_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Ernie4_5_VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    """Supported types:
    - list[`torch.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `torch.Tensor`: A tensor holding all videos' features
        (concatenation of all videos' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on 
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2VLVideoInputs = Union[Ernie4_5_VLVideoPixelInputs,
                           Ernie4_5_VLVideoEmbeddingInputs]

# 视频输入和图片输入相同
Ernie4_5_VLVideoInputs = Union[Ernie4_5_VLImagePixelInputs,
                           Ernie4_5_VLImageEmbeddingInputs]




class TokenType:
    """token type definition"""

    text = 0
    image = 1
    video = 2


class VariableResolutionResamplerModel(nn.Module):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config, prefix: str = "",):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
                self.in_dim
                * self.spatial_conv_size
                * self.spatial_conv_size
                * self.temporal_conv_size
        )

        self.spatial_linear1 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.spatial_linear1",
        )

        # GELU 激活函数
        self.spatial_gelu = nn.GELU()

        self.spatial_linear2 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.spatial_linear2",
        )

        self.spatial_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        if self.use_temporal_conv:
            self.temporal_linear1 = ColumnParallelLinear(
                self.temporal_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, 'quant_config', None),
                prefix=f"{prefix}.temporal_linear1",
            )

            self.temporal_gelu = nn.GELU()

            self.temporal_linear2 = ColumnParallelLinear(
                self.spatial_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, 'quant_config', None),
                prefix=f"{prefix}.temporal_linear2",
            )

            self.temporal_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        self.mlp = ColumnParallelLinear(
            self.spatial_dim,
            self.out_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.mlp",
        )

        self.after_norm = RMSNorm(
            hidden_size=out_dim,
            eps=getattr(config, 'rms_norm_eps', 1e-6)
        )

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size ** 2)])
        return x

    # def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
    def forward(self, x, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        # assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x, _ = self.spatial_linear1(x)
            x = self.spatial_gelu(x)
            x, _ = self.spatial_linear2(x)
            x = self.spatial_norm(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """

            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size ** 2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size ** 2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                    self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                    grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to(
                x.device
            )

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                    grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                        1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to(
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x, _ = self.temporal_linear1(x)
            x = self.temporal_gelu(x)
            x, _ = self.temporal_linear2(x)
            x = self.temporal_norm(x)
            return x

        def fwd_mlp(x):
            x, _ = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x


    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Resampler 权重名称映射表：Sequential层结构 -> 独立层结构
        resampler_weight_mappings = {
            "spatial_linear.0.": "spatial_linear1.",
            "spatial_linear.2.": "spatial_linear2.",
            "spatial_linear.1.": "spatial_norm.",
            "spatial_linear.3.": "spatial_norm.",
            "temporal_linear.0.": "temporal_linear1.",
            "temporal_linear.2.": "temporal_linear2.",
            "temporal_linear.1.": "temporal_norm.",
            "temporal_linear.3.": "temporal_norm.",
        }
        # resampler_params_dict = dict(self.resampler_model.named_parameters())

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            mapped_name = name
            for old_pattern, new_pattern in resampler_weight_mappings.items():
                if old_pattern in name:
                    mapped_name = name.replace(old_pattern, new_pattern)
                    break

            if mapped_name not in params_dict:
                # print(f"=============Warning: {name} is not found in the model.")
                continue
            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)
            # loaded_params.add(f"resampler_model.{mapped_name}")
        return loaded_params



class Ernie4_5_VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        # return self.ctx.get_hf_config(Ernie4_5_VLMoEConfig)
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs: object) -> Ernie_45T_VLProcessor:
        return self.ctx.get_hf_processor(Ernie_45T_VLProcessor,
                                        #  use_fast=True, # TODO 这是的作用是什么
                                         **kwargs)


    def _get_image_processor_kwargs(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        mm_config = self.ctx.model_config.get_multimodal_config()
        if mm_config.mm_processor_kwargs:
            kwargs.update(mm_config.mm_processor_kwargs)

        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels

            if size is None:
                size = {"shortest_edge": min_pixels}
            else:
                size["shortest_edge"] = min_pixels

        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels

            if size is None:
                size = {"longest_edge": max_pixels}
            else:
                size["longest_edge"] = max_pixels

        if size is not None:
            kwargs["size"] = size

        return kwargs

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Ernie_45T_VLImageProcessor:
        return cached_image_processor_from_config(
            self.ctx.model_config,
            **self._get_image_processor_kwargs(min_pixels=min_pixels,
                                               max_pixels=max_pixels,
                                               size=size,
                                               **kwargs),
        )


    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Ernie_45T_VLProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        
        patch_size = vision_config.patch_size
        # merge_size = vision_config.spatial_merge_size
        spatial_conv_size = hf_config.spatial_conv_size
        temporal_conv_size = hf_config.temporal_conv_size
        
        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * spatial_conv_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
                # max_pixels=28 * 28 * 1280,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)
        # TODO 修改为ernie实现
        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        # padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(num_frames // temporal_conv_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (spatial_conv_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Ernie_45T_VLProcessor],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Optional[Ernie_45T_VLProcessor],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens


    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_image_tokens = self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )
        return num_image_tokens



    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        # TODO 如果帧数是奇数，舍弃一帧
        if num_frames % 2 != 0:
            num_frames -= 1

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )



class Ernie4_5VLMultiModalProcessor(BaseMultiModalProcessor[Ernie4_5_VLProcessingInfo]):
    def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        
        """call hf processor"""
        if "images" not in mm_data:
            mm_data["images"] = []
        if "videos" not in mm_data:
            mm_data["videos"] = []

        # 如果mm_data为空并且有placeholder，直接过 tokenizer，不过processor了，不然会报错
        if len(mm_data.get("images", [])) == 0 and len(mm_data.get("videos", [])) == 0 and prompt != "":
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt)
            tokenizer_output = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            return tokenizer_output
        
        images_len = len(mm_data.get("images", []))
        # logger.info(f"  call_hf_processor 的 prompt= {prompt} len(images)={images_len}")
        
        processor_output = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=[prompt], images=mm_data["images"], videos=mm_data["videos"]),
            mm_kwargs,
        )

        # TODO 将输出结果分为两个模态的pixel_values和grid_thw

        
        # TODO 对processor_output进行后处理
        if processor_output is not None:
            for key in list(processor_output.keys()):
                if processor_output[key] is None:
                    del processor_output[key] # 删除空的内容
                    continue
                # 修改key images为pixel_values
                if key == "images":
                    processor_output['pixel_values'] = processor_output['images']
                    # TODO 必须copy一份作为视频模态，不然后面会报错，更好的方法是从pixel_values和grid_thw中分出视频和图片
                    processor_output['pixel_values_videos'] = processor_output['images']
                    del processor_output['images']
                if key == "grid_thw":
                    grid_thw = processor_output['grid_thw']
                    # 找出第一维大于 1 的元素
                    mask = grid_thw[:, 0] > 1
                    # processor_output['grid_thw_video'] = processor_output['grid_thw']
                    processor_output["video_grid_thw"] = grid_thw[mask]
                    processor_output["image_grid_thw"] = grid_thw[~mask]
                if key == "position_ids":
                    processor_output['video_position_ids'] = processor_output['position_ids']



        return processor_output



    # def _cached_apply_hf_processor(
    #         self,
    #         prompt: Union[str, list[int]],
    #         mm_data_items: MultiModalDataItems,
    #         hf_processor_mm_kwargs: Mapping[str, object],
    #         *,
    #         return_mm_hashes: bool,
    # ) -> tuple[list[int], MultiModalKwargs, Optional[dict], bool]:

    #     if mm_data_items.get_count("image", strict=False) > 1:
    #         return self._apply_hf_processor(
    #             prompt=prompt,
    #             mm_data_items=mm_data_items,
    #             hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #             return_mm_hashes=return_mm_hashes,
    #         )
    #     (
    #         prompt_ids,
    #         mm_kwargs,
    #         mm_hashes,
    #         _,
    #     ) = super()._cached_apply_hf_processor(
    #         prompt=prompt,
    #         mm_data_items=mm_data_items,
    #         hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #         return_mm_hashes=True,
    #     )

    #     return prompt_ids, mm_kwargs, mm_hashes, False


    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        # vocab = tokenizer.get_vocab()

        # 被替换的占位符
        before_placeholder = {
            # "image": hf_processor.image_patch_id,
            "image": "<|image@placeholder|>",
            "video": "<|video@placeholder|>"
        }

        # 替换成的占位符
        after_placeholder = {
            "image": "<|IMAGE_PLACEHOLDER|>",
            "video": "<|IMAGE_PLACEHOLDER|>"
        }


        merge_length = hf_processor.spatial_conv_size**2

        def get_replacement_ernie45vl(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            if modality == "video":
                num_tokens = int(grid_thw.prod()) // hf_processor.temporal_conv_size // merge_length
            else:
                num_tokens = int(grid_thw.prod()) // merge_length
            return after_placeholder[modality] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=before_placeholder[modality],
                replacement=partial(get_replacement_ernie45vl,
                                    modality=modality),
            ) for modality in ("image", "video")
            # ) for modality in ("image",)
        ]


    def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:

        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)
        
        return dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_grid_sizes),
                image_grid_thw=MultiModalFieldConfig.batched("image"),
                # position_ids=MultiModalFieldConfig.batched("image"),

                # image_type_ids=MultiModalFieldConfig.batched("video"),
                # input_ids=MultiModalFieldConfig.batched("image"),
                # token_type_ids=MultiModalFieldConfig.batched("image"),

                # ernie45 vl 模型将视频处理成了图片，这里是无效的，仅仅为了过vllm的校验，视频在上面的image中
                pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_grid_sizes),
                # video_embeds=MultiModalFieldConfig.flat_from_sizes(
                #     "video", video_grid_sizes),
                video_grid_thw=MultiModalFieldConfig.batched("video"),
                # position_ids_video=MultiModalFieldConfig.batched("video"),
            )



class Ernie4_5_VLDummyInputsBuilder(BaseDummyInputsBuilder[Ernie4_5_VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        hf_processor = self.info.get_hf_processor()
        # image_placeholder: str = hf_processor.image_placeholder
        # prompt = "<|begin_of_sentence|>User: 描述文件内容是什么"
        prompt = ""
        for i in range(num_images):
            prompt += f"Picture {i+1}:<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"

        for i in range(num_videos):
            prompt += f"Video {i+1}:<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"
        return prompt

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(width=target_width,
                                   height=target_height,
                                   num_frames=target_num_frames,
                                   num_videos=num_videos)
        }




@MULTIMODAL_REGISTRY.register_processor(Ernie4_5VLMultiModalProcessor,
                                        info=Ernie4_5_VLProcessingInfo,
                                        dummy_inputs=Ernie4_5_VLDummyInputsBuilder)
class Ernie4_5_VLMoeForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):
    
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
        # TODO 这里有点trick，上一个model.resampler_model.-> language_model.model.resampler_model. 然后再 language_model.model.resampler_model. -> resampler_model.
        "language_model.model.resampler_model.": "resampler_model.", 
    })
    

    # Resampler 权重名称映射表：Sequential层结构 -> 独立层结构
    _resampler_weight_mappings = {
        "spatial_linear.0.": "spatial_linear1.",
        "spatial_linear.2.": "spatial_linear2.",
        "spatial_linear.1.": "spatial_norm.",
        "spatial_linear.3.": "spatial_norm.",
        "temporal_linear.0.": "temporal_linear1.",
        "temporal_linear.2.": "temporal_linear2.",
        "temporal_linear.1.": "temporal_norm.",
        "temporal_linear.3.": "temporal_norm.",
    }

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config


        # TODO 根据vllm的，这里确实一些参数如 quant_config
        self.vision_model = Ernie4_5_VisionTransformer(vllm_config, prefix=f"vision_model")

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Ernie4_5_VLForCausalLM"],
        )


        self.resampler_model = VariableResolutionResamplerModel(
            self.config.pixel_hidden_size,
            self.config.hidden_size,
            self.config.spatial_conv_size,
            self.config.temporal_conv_size,
            config=self.config,
            prefix=maybe_prefix(prefix, "language_model")
        )
        
        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self._add_image_preprocess(vllm_config)

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """compute logits"""
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)


    def _add_image_preprocess(self, vllm_config):
        processor = AutoProcessor.from_pretrained(vllm_config.model_config.model, ignore_mismatched_sizes=True, trust_remote_code=True)
        device = vllm_config.device_config.device
        image_preprocess = processor.image_processor

        image_preprocess.image_mean_tensor = torch.tensor(
            image_preprocess.image_mean,
            dtype=torch.float32,
            device=device
        ).reshape([1, 3, 1, 1])

        image_preprocess.image_std_tensor = torch.tensor(
            image_preprocess.image_std,
            dtype=torch.float32,
            device=device
        ).reshape([1, 3, 1, 1])

        image_preprocess.rescale_factor = torch.tensor(
            image_preprocess.rescale_factor,
            dtype=torch.float32,
            device=device
        )

        # 硬编码的 patch_size^2
        patch_size_squared = 14 ** 2

        image_preprocess.image_mean_tensor = (
            image_preprocess.image_mean_tensor
            .squeeze([-2, -1])
            .repeat_interleave(patch_size_squared, -1)
        )

        image_preprocess.image_std_tensor = (
            image_preprocess.image_std_tensor
            .squeeze([-2, -1])
            .repeat_interleave(patch_size_squared, -1)
        )

        if not image_preprocess.image_mean_tensor.is_contiguous():
            image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.contiguous()
        if not image_preprocess.image_std_tensor.is_contiguous():
            image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.contiguous()

        self.image_preprocess = image_preprocess

    def _vision_forward(
            self,
            pixel_values,
            grid_thw,
    ):
        """_vision_forward"""
        if self.image_preprocess is not None:
            assert pixel_values.dtype == torch.bfloat16, pixel_values.dtype
            current_device = pixel_values.device
            self.image_preprocess.image_mean_tensor = (
                self.image_preprocess.image_mean_tensor.to(current_device)
            )
            self.image_preprocess.image_std_tensor = (
                self.image_preprocess.image_std_tensor.to(current_device)
            )
            pixel_values = self.image_preprocess.rescale_factor * pixel_values.to(torch.float32)
            pixel_values = (
                             pixel_values - self.image_preprocess.image_mean_tensor
                     ) / self.image_preprocess.image_std_tensor
            pixel_values = pixel_values.to(torch.bfloat16)
        else:
            assert pixel_values.dtype == torch.bfloat16, pixel_values.dtype

        # TODO 这里的作用是什么
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_features = self.vision_model(pixel_values, grid_thw)
        return image_features

    def _merge_multimodal_embeddings(
            self,
            input_ids,
            inputs_embeds,
            image_features,
    ):
        """_merge_multimodal_embeddings"""
        image_mask = input_ids == self.config.im_patch_id

        inputs_embeds[image_mask] = image_features

        return inputs_embeds




    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if getattr(self.config, "im_patch_id", None) is not None:
            self.visual_token_mask = (input_ids == self.config.im_patch_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None
        # logger.info(f"  _set_visual_token_mask 成功 self.visual_token_mask.shape: {self.visual_token_mask.shape}")

    def _set_position_ids(self, position_ids: torch.Tensor) -> None:
        self.position_ids = position_ids
        logger.info(f"  _set_position_ids 成功 self.self.position_ids.shape: {self.position_ids.shape}")

    def get_language_model(self) -> torch.nn.Module:
        """
        返回底层的语言模型

        Returns:
            torch.nn.Module: ERNIE 语言模型
        """
        return self.language_model



    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Ernie4_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Ernie4_5_VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        # if image_embeds is not None:
        #     image_embeds = self._validate_and_reshape_mm_tensor(
        #         image_embeds, "image embeds")
        #     grid_thw = self._validate_and_reshape_mm_tensor(
        #         grid_thw, "image grid_thw")

        #     if not isinstance(image_embeds, torch.Tensor):
        #         raise ValueError("Incorrect type of image embeddings. "
        #                          f"Got type: {type(image_embeds)}")
        #     return Ernie4_5_VLImageEmbeddingInputs(type="image_embeds",
        #                                        image_embeds=image_embeds,
        #                                        grid_thw=grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Ernie4_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Ernie4_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        # if video_embeds is not None:
        #     video_embeds = self._validate_and_reshape_mm_tensor(
        #         video_embeds, "video embeds")
        #     video_grid_thw = self._validate_and_reshape_mm_tensor(
        #         video_grid_thw, "video grid_thw")

        #     if not isinstance(video_embeds, torch.Tensor):
        #         raise ValueError("Incorrect type of video embeddings. "
        #                          f"Got type: {type(video_embeds)}")
        #     return Ernie4_5_VLVideoEmbeddingInputs(type="video_embeds",
        #                                        video_embeds=video_embeds,
        #                                        video_grid_thw=video_grid_thw)

    def _process_image_input(
            self, image_input: Ernie4_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2


        pixel_values = image_input["pixel_values"].type(self.vision_model.dtype)
        image_features = self._vision_forward(pixel_values=pixel_values, grid_thw=grid_thw)
        image_embeds = self.resampler_model(image_features, grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.vision_model.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())
        
    def _process_video_input(
            self, video_input: Ernie4_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        
        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.vision_model.dtype)
        video_features = self._vision_forward(pixel_values=pixel_values_videos, grid_thw=grid_thw)
        video_embeds = self.resampler_model(video_features, grid_thw)

        # Split concatenated embeddings for each video item.

        merge_size = self.vision_model.spatial_merge_size
        # TODO resampler_model之后 维度变为了 grid_thw.prod(-1) // 2
        sizes = (grid_thw.prod(-1) // self.config.temporal_conv_size) // merge_size // merge_size # TODO 修改为 // merge_size**2

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities



    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddi ngs is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings


    def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is None:
            return inputs_embeds

        self._set_visual_token_mask(input_ids)
        inputs_embeds = merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            [self.config.im_patch_id]
        )
        return inputs_embeds

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ):

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # logger.info(f"  self.language_model.model的输入 forward_kwargs: {forward_kwargs} ")
        # logger.info(f"  self.language_model.model的输入 inputs_embeds.shape: {inputs_embeds.shape} positions.shape: {positions.shape} ")
        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            # TODO 这样用会有并发问题吗，internvl是这样用的
            # logger.info(f"  使用self.visual_token_mask，使用后置为None self.visual_token_mask.shape: {self.visual_token_mask.shape}")
            
            if self.visual_token_mask.shape[0] != inputs_embeds.shape[0]:
                logger.warning(f"  self.visual_token_mask.shape[0] != inputs_embeds.shape[0] {self.visual_token_mask.shape}, {inputs_embeds.shape}")
                padding_len = inputs_embeds.shape[0] - self.visual_token_mask.shape[0]
                # 右填充 False
                pad = torch.zeros((padding_len, self.visual_token_mask.shape[1]), dtype=self.visual_token_mask.dtype, device=self.visual_token_mask.device)
                self.visual_token_mask = torch.cat([self.visual_token_mask, pad], dim=0)
            
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(
            **forward_kwargs,
            **kwargs,
        )

        return hidden_states


    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)



# def get_mm_mapping(self) -> MultiModelKeys:
#         """
#         Get the module prefix in multimodal models
#         """
#         # TODO 具体的k-v是什么要根据eb修改
#         return MultiModelKeys.from_string_field(
#             language_model="language_model",
#             connector="visual.merger.",
#             tower_model="visual.",
#         )