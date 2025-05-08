# https://github.com/vllm-project/vllm/blob/5b8c390747a13dde7665e404ee0c4f67270be2f0/vllm/usage/usage_lib.py#L177

import datetime
import json
import logging
import os
import platform
import time
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Union
from uuid import uuid4

import cpuinfo
import psutil
import requests
import torch

import vllm
import vllm.usage
from vllm.usage.usage_lib import UsageContext, UsageMessage
import vllm.usage.usage_lib
from vllm.usage.usage_lib import _detect_cloud_provider, _get_current_timestamp_ns, _USAGE_ENV_VARS_TO_COLLECT
from vllm.version import __version__ as VLLM_VERSION
import vllm.envs as envs

def _report_usage_once(self, model_architecture: str,
                        usage_context: UsageContext,
                        extra_kvs: dict[str, Any]) -> None:
    import torch_xla
    self.gpu_count = torch_xla.runtime.world_size()
    self.gpu_type = torch_xla.tpu.get_tpu_type()
    self.gpu_memory_per_device = (
        torch_xla.core.xla_model.get_memory_info()["bytes_limit"])
    self.provider = _detect_cloud_provider()
    self.architecture = platform.machine()
    self.platform = platform.platform()
    self.total_memory = psutil.virtual_memory().total

    info = cpuinfo.get_cpu_info()
    self.num_cpu = info.get("count", None)
    self.cpu_type = info.get("brand_raw", "")
    self.cpu_family_model_stepping = ",".join([
        str(info.get("family", "")),
        str(info.get("model", "")),
        str(info.get("stepping", ""))
    ])

    # vLLM information
    self.context = usage_context.value
    self.vllm_version = VLLM_VERSION
    self.model_architecture = model_architecture

    # Environment variables
    self.env_var_json = json.dumps({
        env_var: getattr(envs, env_var)
        for env_var in _USAGE_ENV_VARS_TO_COLLECT
    })

    # Metadata
    self.log_time = _get_current_timestamp_ns()
    self.source = envs.VLLM_USAGE_SOURCE

    data = vars(self)
    if extra_kvs:
        data.update(extra_kvs)

    self._write_to_file(data)
    self._send_to_server(data)


vllm.usage.usage_lib.UsageMessage._report_usage_once = _report_usage_once