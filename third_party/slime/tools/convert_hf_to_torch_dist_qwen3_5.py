"""Local wrapper for converting Qwen3.5 checkpoints.

transformers 4.57.1 (pinned by sglang 0.5.9) does not know model_type=qwen3_5
(first shipped in transformers 5.2.0). sglang vendors the config class and
registers it with AutoConfig when imported, which is why training works; the
convert tool's import chain never imports sglang, so mbridge's
AutoBridge.from_pretrained fails there. Register the vendored config here,
then run the standard tool.
"""

import os
import runpy

from transformers import AutoConfig

from sglang.srt.configs.qwen3_5 import Qwen3_5Config

try:
    AutoConfig.register("qwen3_5", Qwen3_5Config)
except ValueError:
    pass  # already registered

runpy.run_path(os.path.join(os.path.dirname(__file__), "convert_hf_to_torch_dist.py"), run_name="__main__")
