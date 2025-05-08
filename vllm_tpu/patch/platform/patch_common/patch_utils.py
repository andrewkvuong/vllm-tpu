# https://github.com/vllm-project/vllm/blob/5b8c390747a13dde7665e404ee0c4f67270be2f0/vllm/model_executor/utils.py#L42

from typing import Any, Dict, Optional

import torch
import vllm

def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    print("Using patched utils!")
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        if key == "weight_loader":
            value = vllm.model_executor.utils._make_synced_weight_loader(value)
        setattr(weight, key, value)


print("DEBUG: About to apply monkey patch for set_weight_attrs.", flush=True)
# Store the original for inspection or if you need to call it
original_set_weight_attrs = vllm.model_executor.utils.set_weight_attrs

vllm.model_executor.utils.set_weight_attrs = set_weight_attrs # Apply your function

print("DEBUG: Monkey patch for set_weight_attrs APPLIED.", flush=True)
print(f"DEBUG: Original set_weight_attrs id: {id(original_set_weight_attrs)}", flush=True)
print(f"DEBUG: Current vllm.model_executor.utils.set_weight_attrs id: {id(vllm.model_executor.utils.set_weight_attrs)}", flush=True)
print(f"DEBUG: Your patched function id: {id(set_weight_attrs)}", flush=True)
#vllm.model_executor.utils.set_weight_attrs = set_weight_attrs
