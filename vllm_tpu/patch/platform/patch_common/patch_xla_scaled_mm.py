# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py

from typing import Optional, Type
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (ScaledMMLinearKernel,
                                   ScaledMMLinearLayerConfig)
from vllm_tpu.model_executor.layers.quantization.kernels.scaled_mm.xla import XLAScaledMMLinearKernel
import vllm
import vllm.model_executor.layers.quantization.kernels.scaled_mm

def choose_scaled_mm_linear_kernel(
        config: ScaledMMLinearLayerConfig,
        compute_capability: Optional[int] = None
) -> Type[ScaledMMLinearKernel]:
    return XLAScaledMMLinearKernel

vllm.model_executor.layers.quantization.kernels.scaled_mm.choose_scaled_mm_linear_kernel = choose_scaled_mm_linear_kernel