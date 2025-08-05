# coding=utf-8
"""
ErnieVL MOE
"""
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, MRotaryEmbedding, get_rope, _apply_rotary_emb_torch, _apply_rotary_emb
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)



from .ernie45_moe import (Ernie4_5_MoeAttention,Ernie4_5_MoeMLP)                                            
from dataclasses import dataclass
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):
    """Original 3D rotary positional embedding. 3D is t:time h:height w:width"""

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            
            section_h = self.mrope_section[0] # 22
            section_w = self.mrope_section[1] # 22
            section_t = self.mrope_section[2] # 20
            assert section_h == section_w
            # 按照 [T, H, W] 拆分
            section_cos_t, section_cos_h, section_cos_w = cos[..., -section_t :], \
                                                            cos[..., : section_h + section_w : 2], \
                                                            cos[..., 1 : section_h + section_w : 2], 
            cos_t, cos_h, cos_w = section_cos_t[0], section_cos_h[1], section_cos_w[2]
            cos_hw = torch.stack([cos_h, cos_w], dim=-1).reshape(cos_h.shape[:-1] + (cos_h.shape[-1] * 2,))
            cos = torch.cat([cos_hw, cos_t], dim=-1)
            
            section_sin_t, section_sin_h, section_sin_w = sin[..., -section_t :], \
                                                            sin[..., : section_h + section_w : 2], \
                                                            sin[..., 1 : section_h + section_w : 2], 
            sin_t, sin_h, sin_w = section_sin_t[0], section_sin_h[1], section_sin_w[2]
            sin_hw = torch.stack([sin_h, sin_w], dim=-1).reshape(sin_h.shape[:-1] + (sin_h.shape[-1] * 2,))
            sin = torch.cat([sin_hw, sin_t], dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key



class Ernie4_5_VLMLP(Ernie4_5_MoeMLP):
    pass


class Ernie4_5_VLAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 500000,
        rope_scaling: Optional[dict[str, Any]] = None,
        freq_allocation: int = 20,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix) if len(prefix) > 0 else 0
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        # self.rotary_emb = get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=max_position_embeddings,
        #     # max_position_embeddings=max_position_embeddings,
        #     base=rope_theta,
        #     is_neox_style=False,
        #     rope_scaling=rope_scaling,
        #     # dtype = torch.get_default_dtype()
        # )

        t_repo = freq_allocation
        h_repo = (self.head_dim // 2 - freq_allocation) // 2
        w_repo = (self.head_dim // 2 - freq_allocation) // 2

        self.rotary_emb = Ernie4_5_VLRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype = torch.get_default_dtype(),
            mrope_section=[h_repo, w_repo , t_repo]
            )


        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v)
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output



class Ernie4_5_VLMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.has_shared_experts = (getattr(config, "moe_num_shared_experts", 0)
                                   > 0)
        self.hidden_size = config.hidden_size

        moe_num_experts = getattr(config, "moe_num_experts", 0)
        if isinstance(moe_num_experts, list):
            max_moe_num_experts = max(moe_num_experts)
        else:
            max_moe_num_experts = moe_num_experts

        if self.tp_size > max_moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {moe_num_experts}.")


        

        moe_layer_start_index = config.moe_layer_start_index
        if isinstance(moe_layer_start_index, int):
            text_moe_layer_start_index = moe_layer_start_index
            image_moe_layer_start_index = moe_layer_start_index
        else:
            text_moe_layer_start_index = moe_layer_start_index[0]
            image_moe_layer_start_index = moe_layer_start_index[1]
        
        moe_layer_end_index = config.moe_layer_end_index
        if moe_layer_end_index is None:
            text_moe_layer_end_index = config.num_layers
            image_moe_layer_end_index = config.num_layers
        elif isinstance(moe_layer_end_index, int):
            text_moe_layer_end_index = moe_layer_end_index
            image_moe_layer_end_index = moe_layer_end_index
        else:
            text_moe_layer_end_index = moe_layer_end_index[0]
            image_moe_layer_end_index = moe_layer_end_index[1]
        assert text_moe_layer_start_index <= text_moe_layer_end_index
        
        
        if layer_idx >= text_moe_layer_start_index and layer_idx <= text_moe_layer_end_index:
            self.text_experts_gate = ReplicatedLinear(config.hidden_size,
                        config.moe_num_experts[0],
                        bias=False,
                        quant_config=quant_config,
                                prefix=f"{prefix}.gate")
            # TODO 检查这里的入参
            self.text_experts = FusedMoE(num_experts=config.moe_num_experts[0],
                                top_k=config.moe_k,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size[0],
                                reduce_results=False,
                                renormalize=True,
                                # expert_id_offset=0, # TODO 明确这个含义，vllm中没有，fd中有，猜测是用于加载权重用的？
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")
            # TODO 这里还缺少bias
        else:
            # TODO 检查这里的入参
            self.text_experts = Ernie4_5_VLMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, 'use_bias', False),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")

        assert image_moe_layer_start_index <= image_moe_layer_end_index
        if layer_idx >= image_moe_layer_start_index and layer_idx <= image_moe_layer_end_index:
            self.image_experts_gate = ReplicatedLinear(config.hidden_size,
                        config.moe_num_experts[1],
                        bias=False,
                        quant_config=quant_config,
                                prefix=f"{prefix}.gate")
            # TODO 检查这里的入参
            self.image_experts = FusedMoE(num_experts=config.moe_num_experts[1],
                                top_k=config.moe_k,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size[1],
                                reduce_results=False,
                                renormalize=True,
                                # expert_id_offset=config.moe_num_experts[0], # TODO 明确这个含义，vllm中没有，fd中有，猜测是用于加载权重用的？
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")
            # TODO 这里还缺少bias
        else:
            # TODO 检查这里的入参
            self.image_experts = Ernie4_5_VLMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, 'use_bias', False),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")

        # self.experts = FusedMoE(num_experts=config.moe_num_experts,
        #                         top_k=config.moe_k,
        #                         hidden_size=config.hidden_size,
        #                         intermediate_size=config.moe_intermediate_size,
        #                         reduce_results=False,
        #                         renormalize=True,
        #                         quant_config=quant_config,
        #                         prefix=f"{prefix}.experts")

        if self.has_shared_experts:
            intermediate_size = (config.moe_intermediate_size[0] *
                                 config.moe_num_shared_experts)
            self.shared_experts = Ernie4_5_VLMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=self.text_experts.must_reduce_shared_expert_outputs(
                )) # TODO 这里self.experts到底要用哪个专家呢

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        **kwargs: object,
        ) -> torch.Tensor:
        
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        

        if self.has_shared_experts:
            shared_output = self.shared_experts(hidden_states)

        
        if visual_token_mask is not None and visual_token_mask.any():
            # assert visual_token_mask.shape[0] != hidden_states.shape[0]
            visual_token_mask = visual_token_mask.repeat(
                1, self.hidden_size).bool()
            text_token_mask =  ~visual_token_mask
            final_hidden_states = torch.zeros_like(hidden_states)
            
            text_hidden_states = hidden_states[text_token_mask].reshape(-1, self.hidden_size)
            image_hidden_states = hidden_states[visual_token_mask].reshape(-1, self.hidden_size)

            text_router_logits, _ = self.text_experts_gate(text_hidden_states)
            final_hidden_states[text_token_mask] = self.text_experts(hidden_states=text_hidden_states,
                                                                    router_logits=text_router_logits).flatten()
            
            image_router_logits, _ = self.image_experts_gate(image_hidden_states)
            final_hidden_states[visual_token_mask]  = self.image_experts(hidden_states=image_hidden_states,
                                                                    router_logits=image_router_logits).flatten()
        else:
            # 单模态输入直接处理
            text_router_logits, _ = self.text_experts_gate(hidden_states)

            final_hidden_states = self.text_experts(hidden_states=hidden_states,
                                           router_logits=text_router_logits)

        if self.has_shared_experts and \
              shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            # TODO 这里self.experts到底要用哪个专家呢
            final_hidden_states = (
                self.text_experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states))

        return final_hidden_states.view(orig_shape)




class Ernie4_5_VLDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 500000)
        rope_scaling = getattr(config, "rope_scaling", None)
        freq_allocation = getattr(config, "freq_allocation", 20)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          131072)
        # TODO 检查attention
        self.self_attn = Ernie4_5_VLAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            freq_allocation=freq_allocation,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'use_bias', False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        # MoE
        moe_layer_start_index = getattr(config, "moe_layer_start_index", 0)
        if isinstance(moe_layer_start_index, list):
            min_moe_layer_start_index = min(moe_layer_start_index)
        else:
            min_moe_layer_start_index = moe_layer_start_index
        
        moe_layer_end_index = getattr(config, "moe_layer_end_index",
                                      config.num_hidden_layers - 1)
        if isinstance(moe_layer_end_index, list):
            max_moe_layer_end_index = max(moe_layer_end_index)
        else:
            max_moe_layer_end_index = moe_layer_end_index

        assert min_moe_layer_start_index <= max_moe_layer_end_index

        moe_num_experts = getattr(config, "moe_num_experts", 0)
        if isinstance(moe_num_experts, list):
            max_moe_num_experts = max(moe_num_experts)
        else:
            max_moe_num_experts = moe_num_experts

        moe_layer_interval = getattr(config, "moe_layer_interval", 1)
        use_moe = getattr(config, "use_moe", max_moe_num_experts > 0)

        if (use_moe and ((layer_idx + 1) % moe_layer_interval == 0)
                and layer_idx >= min_moe_layer_start_index
                and layer_idx <= max_moe_layer_end_index):
            self.mlp = Ernie4_5_VLMoE(config=config,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.mlp")
        else:
            self.mlp = Ernie4_5_VLMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                use_bias=getattr(config, 'use_bias', False),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        visual_token_mask: Optional[torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        
        if isinstance(self.mlp, Ernie4_5_VLMoE):
            hidden_states = self.mlp(hidden_states, visual_token_mask, **kwargs)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# TODO doing
# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         "positions": -1,
#         "intermediate_tensors": 0,
#         "inputs_embeds": 0,
#         "visual_token_mask": 0,
#     })
class Ernie4_5_VLModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.im_patch_id = config.im_patch_id

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Ernie4_5_VLDecoderLayer(config=config,
                                                    cache_config=cache_config,
                                                    quant_config=quant_config,
                                                    prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        visual_token_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual, visual_token_mask, **kwargs) # TODO 传入vl_moe_meta

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states



class  Ernie4_5_VLForCausalLM(nn.Module, SupportsPP):
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

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Ernie4_5_VLModel(vllm_config=vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max(self.config.moe_num_experts))

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and name.endswith(
                    "lm_head.weight"):
                loaded_params.add("lm_head.weight")
                continue
            # MTP will be supported soon.
            if "mtp" in name or "vision_model" in name or "resampler_model" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                # print(f"name:{name} loaded_weight shape:{loaded_weight.shape}")
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # TODO 文本专家 和 视觉专家
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    image_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = True if moe_offset <= image_expert_start_idx - 1 else False
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(f".experts.{moe_offset}", f".image_experts.{moe_offset-image_expert_start_idx}")
                
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in name:
                        continue
                    
                    # TODO 判断是 文本专家 还是 视觉专家
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = True if moe_offset <= self.config.moe_num_experts[0] - 1 else False

                    name = name.replace(weight_name, param_name)
                    # 把name中的experts换为text_experts或者image_experts
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".image_experts.")
                    
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]

                    # print(f"name:{name} loaded_weight shape:{loaded_weight.shape}")
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # TODO 文本gate和视觉gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace("gate.weight_1", "image_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    
                    param = params_dict[name]

                    # print(f"name:{name} loaded_weight shape:{loaded_weight.shape}")
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
