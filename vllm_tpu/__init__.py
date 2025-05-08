def register():
    """Register the TPU platform."""
    print("USING VLLM_TPU!!!!")
    return "vllm_tpu.platform.TPUPlatform"


def register_model():
    from .model_executor.models import register_model
    register_model()
