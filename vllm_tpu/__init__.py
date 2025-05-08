def register():
    """Register the TPU platform."""
    from vllm_tpu.patch import platform  # noqa: F401
    return "vllm_tpu.platform.TpuPlatform"


def register_model():
    from .model_executor.models import register_model
    register_model()
