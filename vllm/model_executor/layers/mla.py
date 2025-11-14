# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from vllm.attention import Attention
from vllm.attention.layers.mla_dp_rebalancing import get_mla_dp_rebalancing_context
from vllm.attention.ops.hpu_shared_weight_layer import (
    post_process_after_loading_for_shared_weight_series,
    reach_layer_for_shared_weight_series,
    register_layer_to_shared_weight_series,
)
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_tp_group,
)
from vllm.distributed.parallel_state import (
    get_mla_dp_rebalancing_o_shared_group,
    get_mla_dp_rebalancing_world_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: Optional[torch.nn.Module]
    kv_a_proj_with_mqa: Optional[torch.nn.Module]
    q_a_layernorm: Optional[torch.nn.Module]
    q_b_proj: Optional[torch.nn.Module]
    q_proj: Optional[torch.nn.Module]
    indexer: Optional[torch.nn.Module]
    is_sparse: bool
    topk_indices_buffer: Optional[torch.Tensor]


@CustomOp.register("multi_head_latent_attention")
class MultiHeadLatentAttention(CustomOp):
    """MLA layer registered as CustomOp.
    Note that currently MLA ignores the enable/disable mechanism of CustomOp
    because there is only one in-tree implementation in forward_native.
    TODO: implement this with a new PluggableLayer mechanism.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        if get_current_vllm_config().parallel_config.enable_mla_prefill_dp_rebalancing:
            # Dispose tensor from the original o_proj
            for attr_name in dir(self.o_proj):
                attr_value = getattr(self.o_proj, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    # logger.debug("***wyt*** mla.py self.o_proj attr: %s", attr_value) # 6 layers 为啥有12项呀。。 
                    # Parameter(ModelWeightParameter([], device='hpu:0', dtype=torch.float8_e4m3fn))
                    attr_value.set_(  # 所以set_是只改storage，不修改metadata的？lazy会出错
                    # attr_value.copy_(
                        torch.empty(
                            (0,), device=attr_value.device, dtype=attr_value.dtype
                        )
                    )
            # Construct the new o_proj using ReplicatedLinear
            # 原先的o_proj是行并行的
            new_o_proj = ReplicatedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=self.o_proj.prefix,
            )
            # Replace the o_proj with the new one
            self.o_proj.__class__ = new_o_proj.__class__
            self.o_proj.__dict__ = new_o_proj.__dict__
            # Register the o_proj into shared weight series to cut down memory usage
            register_layer_to_shared_weight_series(
                series_name="o_proj",
                group=get_mla_dp_rebalancing_o_shared_group(),
                layer=self.o_proj,
                prefetch_step=1,
            )

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            use_sparse=mla_modules.is_sparse,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # 不知道这个函数能不能走到，要验证一下
        # vllm/model_executor/model_loader/utils.py中
        # Complete the initialization of shared weight for o_proj
        if get_current_vllm_config().parallel_config.enable_mla_prefill_dp_rebalancing:
            post_process_after_loading_for_shared_weight_series(self.o_proj)

        logger.debug(
            "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention"
            "enter process_weights_after_loading()"
        )

    def _forward_prefill_with_dp_rebalancing(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.fused_qkv_a_proj is not None, (
            "fused_qkv_a_proj is required when q_lora_rank is not None"
        )
        assert self.q_a_layernorm is not None, (
            "q_a_layernorm is required when q_lora_rank is not None"
        )
        assert self.q_b_proj is not None, (
            "q_b_proj is required when q_lora_rank is not None"
        )
        context = get_mla_dp_rebalancing_context()
        assert context is not None
        # Split inputs from local DP to each device.
        dp_sp_hidden_states = hidden_states
        device_sp_hidden_states = dp_sp_hidden_states[
            context.local_device_sp_start_token_within_dp : context.local_device_sp_end_token_within_dp  # noqa: E501
        ]
        logger.info(
            "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention dp_sp_hidden_states.shape:%s, device_sp_hidden_states.shape:%s", dp_sp_hidden_states.shape, device_sp_hidden_states.shape
        ) # 目前没有维度1了 dp_sp_hidden_states.shape:torch.Size([1, 513, 7168]), device_sp_hidden_states.shape:torch.Size([1, 513, 7168])
        
        # MLA prefill:
        # 1. Perform q_a_proj and q_a_layernorm to obtain q_c
        # 2. Perform kv_a_proj_with_mqa to obtain kv_no_split
        # maybe_npu_prefetch(inputs=self.q_a_proj.weight,
        #                    dependency=hidden_states,
        #                    enabled=self.enable_prefetch)
        # 这里self.fused_qkv_a_proj是将q_down_proj和kv_down_proj合在一起计算的，
        # 但我理解这应该是OK的？
        sp_qkv_lora = self.fused_qkv_a_proj(device_sp_hidden_states)[0]
        sp_q_c, sp_kv_lora = sp_qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        sp_q_c = self.q_a_layernorm(sp_q_c)
        # Rearrange down_proj outputs across DP.
        logger.info(
            "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention sp_q_c.shape:%s, sp_kv_lora:%s", sp_q_c.shape, sp_kv_lora.shape
        )  # sp_q_c.shape:torch.Size([1, 513, 1536]), sp_kv_lora:torch.Size([1, 513, 576])
        sp_down_proj_output = torch.cat([sp_q_c, sp_kv_lora], dim=-1)  # [1, 513, 1536+576] = [1, 513, 2112]
        sp_world_group = get_mla_dp_rebalancing_world_group()
        global_sp_down_proj_output = sp_world_group.all_gather(sp_down_proj_output, 0)
        local_dp = context.local_dp
        dp_ori_down_proj_output = global_sp_down_proj_output[
            context.start_token_of_dp[local_dp] : context.end_token_of_dp[local_dp]
        ]
        prefill_q_c, prefill_kv_no_split = dp_ori_down_proj_output.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        logger.info(
            "***wyt*** mla.py class MultiHeadLatentAttention prefill_q_c.shape:%s, prefill_kv_no_split.shape:%s", prefill_q_c.shape, prefill_kv_no_split.shape
        ) # prefill_q_c.shape:torch.Size([2048, 1536]), prefill_kv_no_split.shape:torch.Size([2048, 576])

        q = self.q_b_proj(prefill_q_c)[0]

        kv_c, k_pe = prefill_kv_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)
        logger.info(
            "***wyt*** mla.py class MultiHeadLatentAttention q.shape:%s, kv_c_normed.shape:%s, k_pe.shape:%s", q.shape, kv_c_normed.shape, k_pe.shape
        ) # q.shape:torch.Size([2048, 128, 192]), kv_c_normed.shape:torch.Size([2048, 512]), k_pe.shape:torch.Size([2048, 1, 64])
        logger.info(
            "***wyt*** mla.py class MultiHeadLatentAttention positions.shape:%s, self.qk_nope_head_dim:%s", positions.shape, self.qk_nope_head_dim
        )  # positions.shape:torch.Size([1, 2080]), self.qk_nope_head_dim:128
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, prefill_q_c, positions, self.rotary_emb
            )

        output_prefill = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        # Rearrange attention outputs across DP to run SP for o_proj.
        tp_size = get_tp_group().world_size
        total_receive_len = context.local_device_total_receive_len
        sp_o_proj_input = torch.empty(
            [total_receive_len * tp_size, self.num_heads * self.v_head_dim],
            dtype=output_prefill.dtype,
            device=output_prefill.device,
        )
        if get_dp_group().world_size == 1:
            if (
                output_prefill.shape[0] < context.num_padded_global_tokens
            ):  # 这个是必要的嘛？
                output_prefill = nn.functional.pad(
                    output_prefill,
                    (
                        0,
                        0,
                        0,
                        context.num_padded_global_tokens - output_prefill.shape[0],
                    ),
                )
            dist.all_to_all_single(
                output=sp_o_proj_input,
                input=output_prefill,
                group=sp_world_group.device_group,
            )
        else:
            dist.all_to_all_single(
                output=sp_o_proj_input,
                input=output_prefill,
                output_split_sizes=context.output_split_sizes,
                input_split_sizes=context.input_split_sizes,
                group=sp_world_group.device_group,
            )
        sp_o_proj_input = sp_o_proj_input.reshape(
            total_receive_len, tp_size * self.num_heads * self.v_head_dim
        )
        if total_receive_len < context.num_tokens_per_device:
            sp_o_proj_input = nn.functional.pad(
                sp_o_proj_input,
                (0, 0, 0, context.num_tokens_per_device - total_receive_len),
            )
        # O proj
        sp_o_proj_output = self.o_proj(sp_o_proj_input)[0]
        del sp_o_proj_input
        # 下面的all_gather是一定需要的嘛？
        # 嗯 RowParallelLinear默认是all-reduce回完整的结果
        return get_tp_group().all_gather(sp_o_proj_output, 0)

    def forward_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )
            forward_context: ForwardContext = get_forward_context()
            is_prefill = forward_context.attn_metadata.is_prompt
            if (
                is_prefill
                and get_current_vllm_config().parallel_config.enable_mla_prefill_dp_rebalancing  # noqa: E501
            ):
                # prefetch the weights
                reach_layer_for_shared_weight_series(self.o_proj)
                if get_mla_dp_rebalancing_context() is not None:
                    return self._forward_prefill_with_dp_rebalancing(
                        positions, hidden_states
                    )
            logger.info(
                "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention hidden_states:%s", hidden_states.shape
            )  # hidden_states:torch.Size([1, 1, 7168])->[1,7168]    [1, 2080, 7168]->[2080, 7168]
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            logger.info(
                "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention q_c.shape:%s, kv_lora.shape:%s", q_c.shape, kv_lora.shape
            ) # q_c.shape:torch.Size([1, 2080, 1536]), kv_lora.shape:torch.Size([1, 2080, 576])
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        logger.info(
            "***wyt*** vllm/model_executor/layers/mla.py class MultiHeadLatentAttention q.shape:%s, kv_lora.shape:%s", q.shape, kv_lora.shape
        ) # q.shape:torch.Size([1, 1, 24576]), kv_lora.shape:torch.Size([1, 1, 576])
        # q.shape:torch.Size([1, 2080, 24576]), kv_lora.shape:torch.Size([1, 2080, 576])
        # 24576 = 128 * (128 + 64) num_heads * head_dim
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(hidden_states, q_c, positions, self.rotary_emb)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        # O proj
        # When o_proj is ReplicatedLinear, make sure the second dimension is
        # complete while decoding.
        # 这里是decode情况下会走到嘛？prefill不是前面走掉了嘛，
        # 所以是decode会走到的？
        # PD分离情况下，不会出现同一批次两者混合情况的。讲道理就是不会走到这儿呀，
        # 因为decode情况不会打开这个环境变量呀
        # if get_current_vllm_config().parallel_config.
        # enable_mla_prefill_dp_rebalancing:
        #     attn_out = get_tp_group().all_gather(attn_out, dim=-1)
        #     output[...] = self.o_proj(attn_out)[0]  # output是未pad的
        #     del attn_out
        #     return output_padded  # 还没改完。。。

        return self.o_proj(attn_out)[0]

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
