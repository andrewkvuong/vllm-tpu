# https://github.com/vllm-project/vllm/blob/6115b115826040ad1f49b69a8b4fdd59f0df5113/vllm/model_executor/model_loader/loader.py#L395
import time
from typing import Generator, Tuple

import torch
import vllm
import vllm.model_executor.model_loader
import vllm.model_executor.model_loader.default_loader
from vllm.config import LoadFormat
from vllm.model_executor.model_loader.default_loader.DefaultModelLoader import \
    Source
from vllm.model_executor.model_loader.weight_utils import (
    fastsafetensors_weights_iterator, np_cache_weights_iterator,
    pt_weights_iterator, safetensors_weights_iterator)


def _get_weights_iterator(
        self,
        source: "Source") -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Get an iterator for the model weights based on the load format."""
    hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
        source.model_or_path, source.revision, source.fall_back_to_pt,
        source.allow_patterns_overrides)
    if self.load_config.load_format == LoadFormat.NPCACHE:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False
        weights_iterator = np_cache_weights_iterator(
            source.model_or_path,
            self.load_config.download_dir,
            hf_folder,
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
        )
    elif use_safetensors:
        if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
            weights_iterator = fastsafetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        else:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
    else:
        weights_iterator = pt_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            self.load_config.pt_load_map_location,
        )

    # In PyTorch XLA, we should call `xm.mark_step` frequently so that
    # not too many ops are accumulated in the XLA program.
    import torch_xla.core.xla_model as xm

    def _xla_weights_iterator(iterator: Generator):
        for weights in iterator:
            yield weights
            xm.mark_step()

    weights_iterator = _xla_weights_iterator(weights_iterator)

    if self.counter_before_loading_weights == 0.0:
        self.counter_before_loading_weights = time.perf_counter()
    # Apply the prefix.
    return ((source.prefix + name, tensor)
            for (name, tensor) in weights_iterator)


vllm.model_executor.model_loader.default_loader.DefaultModelLoader._get_weights_iterator = _get_weights_iterator
