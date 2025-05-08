def register():
    """Register the TPU platform."""

    return "vllm_tpu.platform.TPUPlatform"


def register_model():
    from .model_executor.models import register_model
    register_model()
