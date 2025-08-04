"""
Ernie-45T VIT
"""
# coding=utf-8
import math
import os
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import nn
from transformers import PreTrainedModel

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from functools import partial
from vllm.distributed import utils as dist_utils
from vllm.platforms import _Backend, current_platform
from vllm.attention.layer import MultiHeadAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from .vision import get_vit_attn_backend

logger = init_logger(__name__)


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1),
                         "... d two -> ... (d two)",
                         two=2)


def apply_rotary_emb_torch(x: torch.Tensor,
                           cos: torch.Tensor,
                           sin: torch.Tensor,
                           interleaved: bool = False) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos +
            rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor,
                                freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    apply_rotary_emb = apply_rotary_emb_torch
    if current_platform.is_cuda():
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
    output = apply_rotary_emb(t_, cos, sin).type_as(t)
    return output


class Ernie4_5_VisionAttention(nn.Module):
    """VisionAttention using VLLM framework APIs"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        vllm_config=None,
    ) -> None:
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 计算 tensor parallel 相关的参数
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.num_heads_per_partition = self.num_heads // tp_size if tp_size <= self.num_heads else self.num_heads


        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = world_size
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size)


        # 手动定义 q_size 和 kv_size（类似 Llama 实现）
        self.q_size = self.num_heads_per_partition * self.head_dim
        self.kv_size = self.num_heads_per_partition * self.head_dim  # self-attention
        self.scaling = self.head_dim**-0.5

        self.attn = MultiHeadAttention(self.num_heads_per_partition, self.head_dim,
                                self.scaling)


        # 使用 QKVParallelLinear 替换原始的线性层
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim, # TODO 这里为什么用embed_dim，不用hidden_size呢
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,  # self-attention，kv heads = q heads
            bias=True,
            quant_config=vllm_config.quant_config if vllm_config else None,
            prefix=f"{prefix}.qkv"
        )

        # 使用 RowParallelLinear 替换输出投影层
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=vllm_config.quant_config if vllm_config else None,
            prefix=f"{prefix}.proj"
        )

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"Qwen2-VL does not support {self.attn_backend} backend now.")

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = tensor_model_parallel_all_gather(qkv)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v


    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:

        # [s, b, c] --> [s, b, 3 * head * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.attn_backend == _Backend.FLASH_ATTN:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            from flash_attn import flash_attn_varlen_func

            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                 for x in [q_i, k_i, v_i])
                output_i = F.scaled_dot_product_attention(q_i,
                                                          k_i,
                                                          v_i,
                                                          scale=self.scaling,
                                                          dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None,
                                                       device=q.device)

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=self.scaling)
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output

    def forward_bak(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        """forward function for vision attention"""
        ##############################################################################

        qkv, _ = self.qkv(
            hidden_states
        )  # batch_size, q_len, 3 * num_heads_per_partition * head_dim
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q = q.view(q.shape[0], q.shape[1], self.num_heads_per_partition, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads_per_partition, self.head_dim)
        # q = q.view(q.shape[0], self.num_heads_per_partition, self.head_dim)
        # k = k.view(k.shape[0], self.num_heads_per_partition, self.head_dim)

        # 应用 RoPE
        if rotary_pos_emb is not None:
            # q = apply_rotary_pos_emb_vision(q.unsqueeze(dim=0), rotary_pos_emb)
            # k = apply_rotary_pos_emb_vision(k.unsqueeze(dim=0), rotary_pos_emb)
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        # v = v.unsqueeze(dim=0)


        out = self.attn(q, k, v)
        attn_output, _ = self.proj(out)
        # if attn_output.ndim == 3:
        #     attn_output = attn_output.squeeze(dim=0)
        return attn_output

        ##############################################################################
        # seq_length = hidden_states.shape[0]

        # qkv, _ = self.qkv(
        #     hidden_states
        # )  # batch_size, q_len, 3 * num_heads_per_partition * head_dim
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q = q.view(q.shape[0], self.num_heads_per_partition, self.head_dim)
        # k = k.view(k.shape[0], self.num_heads_per_partition, self.head_dim)
        # v = v.view(k.shape[0], self.num_heads_per_partition, self.head_dim)

        # q = apply_rotary_pos_emb_vision(q.unsqueeze(dim=0), rotary_pos_emb).squeeze(
        #     dim=0
        # )
        # k = apply_rotary_pos_emb_vision(k.unsqueeze(dim=0), rotary_pos_emb).squeeze(
        #     dim=0
        # )
        
        # q = q.transpose(0, 1)
        # k = k.transpose(0, 1)
        # v = v.transpose(0, 1)
        
        # lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # splits = [
        #     torch.split(tensor, lengths.tolist(), dim=1) for tensor in (q, k, v)
        # ]
        
        # attn_output = []
        # for q, k, v in zip(*splits):
        #     # attn_weights = torch.matmul(q, k.transpose(1, 2))
        #     # attn_weights = F.softmax(attn_weights * self.scaling, dim=-1)
        #     # attn_output_splited = torch.matmul(attn_weights, v)
        #     # attn_output_splited = attn_output_splited.transpose(0, 1)
        #     # attn_output.append(attn_output_splited)
        #     output_i = F.scaled_dot_product_attention(q,
        #                                               k,
        #                                               v,
        #                                               scale=self.scaling)
        #     output_i = output_i.transpose(0, 1)
        #     attn_output.append(output_i)
        # attn_output = torch.cat(attn_output, dim=0)
        # attn_output = attn_output.reshape(seq_length, -1).contiguous()
        # attn_output, _ = self.proj(attn_output)
        # return attn_output



class Ernie4_5_VisionMLP(nn.Module):
    """VisionMLP using VLLM parallel linear layers"""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str, vllm_config=None, prefix="") -> None:
        super().__init__()

        # 直接使用传入的参数，而不是从config重新计算
        # 这确保与原始Transformer实现的尺寸一致
        in_features = dim
        hidden_features = hidden_dim

        self.fc1 = ColumnParallelLinear(in_features,
                                        hidden_features,
                                        bias=True,  # 显式设置bias，与原始nn.Linear保持一致
                                        quant_config=vllm_config.quant_config,
                                        prefix=f"{prefix}.fc1")
        self.act = get_act_fn(hidden_act)

        self.fc2 = RowParallelLinear(hidden_features,
                                     in_features,
                                     bias=True,  # 显式设置bias，与原始nn.Linear保持一致
                                     quant_config=vllm_config.quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: VisionMLP output tensor
        """

        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)

        return x


class PatchEmbed(nn.Module):
    """PatchEmbed using VLLM linear layer"""

    def __init__(
            self,
            patch_size: int = 14,
            in_channels: int = 3,
            embed_dim: int = 1152,
            vllm_config=None,
            prefix="",
    ) -> None:
        """
        Args:
            patch_size (int, optional): patch size. Defaults to 14.
            in_channels (int, optional): number of channels. Defaults to 3.
            embed_dim (int, optional): embedding dimension. Defaults to 1152.
            vllm_config: VLLM configuration for quantization support
            prefix: prefix for weight loading
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 使用 ColumnParallelLinear 替换普通的线性层
        self.proj = ColumnParallelLinear(
            input_size=in_channels * patch_size * patch_size,
            output_size=embed_dim,
            bias=False,
            gather_output=True,  # 收集输出，因为这是 patch embedding
            quant_config=vllm_config.quant_config if vllm_config else None,
            prefix=f"{prefix}.proj"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): hidden states

        Returns:
            torch.Tensor: output tensor
        """

        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        hidden_states, _ = self.proj(hidden_states)

        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    """VisionRotaryEmbedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            theta (float, optional): the frequency factor. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim)

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Args:
            seqlen (int): length of sequence.

        Returns:
            torch.Tensor: rotary position embedding
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


class Ernie4_5_VisionBlock(nn.Module):
    """Qwen2VisionBlock"""

    def __init__(self, vllm_config, prefix="", block_idx=None) -> None:
        """
        Args:
            config (dict): model configuration.
            attn_implementation (str, optional): attention implementation. Defaults to "sdpa".
            block_idx (int, optional): block index for debugging control.
        """
        super().__init__()
        config = vllm_config.model_config.hf_config.vision_config
        embed_dim = config.embed_dim
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.block_idx = block_idx

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        # 传递 vllm_config 给 VisionAttention
        self.attn = Ernie4_5_VisionAttention(
            embed_dim=embed_dim,
            num_heads=config.num_heads,
            projection_size=embed_dim,
            vllm_config=vllm_config,
            prefix=f"{prefix}.attn"
        )
        self.attn._debug_block_idx = block_idx

        self.mlp = Ernie4_5_VisionMLP(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act,
                             vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.mlp._debug_block_idx = block_idx

        self.config = config

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Ernie4_5_VisionTransformer(nn.Module):
    """DFNRopeVisionTransformerPreTrainedModel"""

    _tp_plan = {}

    def __init__(self, vllm_config, prefix="") -> None:
        """
        Args:
            config (dict): model configuration
        """
        config = vllm_config.model_config.hf_config.vision_config
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.prefix = prefix
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            vllm_config=vllm_config,
            prefix=f"{prefix}.patch_embed",
        )
        self.quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config
        self.vllm_config = vllm_config

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Ernie4_5_VisionBlock(vllm_config, prefix=f"{prefix}.block_{i}", block_idx=i) for i in range(config.depth)])

        assert (
                config.hidden_size == config.embed_dim
        ), "in DFNRope, vit's config.hidden must be equal to config.embed_dim"
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        rot_pos_emb
        """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb



    def compute_attn_mask_seqlen(
            self, cu_seqlens: torch.Tensor
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_pad=0) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input tensor
            grid_thw (torch.Tensor): grid thw of input
            num_pad (int): number of padding tokens

        Returns:
            torch.Tensor: output tensor
        """


        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)


        # 增加batch size
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(dim=1)


        # pre-compute seqlens for attn mask to reduce cuMemcpy operations
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        for i, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )


        final_output = self.ln(hidden_states)

        if final_output.ndim == 3:
             final_output = final_output.squeeze(dim=1)

        return final_output

    def load_weights(self, weights) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # ("qkv", "q_proj", "q"),
            # ("qkv", "k_proj", "k"),
            # ("qkv", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params