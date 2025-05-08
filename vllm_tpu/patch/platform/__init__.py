from packaging.version import InvalidVersion, Version

def vllm_version_is(target_vllm_version: str):
    import vllm
    vllm_version = vllm.__version__
    try:
        return Version(vllm_version) == Version(target_vllm_version)
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the vaule follows the "
            "format of x.y.z.")

# Import specific patches for different versions
if vllm_version_is("0.8.5") or vllm_version_is("0.8.5.post1"):
    from vllm_tpu.patch.platform import patch_0_8_5  # noqa: F401
    from vllm_tpu.patch.platform import patch_common  # noqa: F401
else:
    from vllm_tpu.patch.platform import patch_common  # noqa: F401
    from vllm_tpu.patch.platform import patch_main  # noqa: F401
