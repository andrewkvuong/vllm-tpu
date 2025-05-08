# https://github.com/vllm-project/vllm/blob/main/vllm/attention/layer.py#L330

import torch
import vllm
import vllm.attention.layer


def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Input shape: batch_size x seq_len x hidden_size"""
    # TODO(Isotr0py): Use existing backend implementations and support FA3
    bsz, q_len, _ = query.size()
    kv_len = key.size(1)

    query = query.view(bsz, q_len, self.num_heads, self.head_size)
    key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
    value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

    if (num_repeat := self.num_queries_per_kv) > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_repeat, dim=2)
        value = torch.repeat_interleave(value, num_repeat, dim=2)

    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    from torch_xla.experimental.custom_kernel import flash_attention
    out = flash_attention(query, key, value, sm_scale=self.scale)
    out = out.transpose(1, 2)

    return out.reshape(bsz, q_len, -1)


vllm.attention.layer.MultiHeadAttention.forward = forward
