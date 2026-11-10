# Training image for Gemma 4 runs. Built FROM the stock slime image so the
# Qwen3.5 campaigns keep using slimerl/slime:latest untouched -- the changes
# here would otherwise have to be applied container-locally on every node and
# reapplied after each create_officeqa_train.sh.
#
# Why any of this is needed: google/gemma-4-12B-it ships as
# `model_type: gemma4_unified`, a format transformers 5.8.1 (the version the
# stock image pins) cannot parse at all, so AutoConfig refuses the checkpoint
# and BOTH the Megatron conversion and the SGLang rollout fail before touching
# weights. gemma-4-26B-A4B-it and gemma-4-E4B-it are plain `gemma4` and need
# none of this.
#
# Build (on the dev box, ~24G model not required):
#   docker build -t officeqa-slime:gemma4 docker/slime-gemma4
#
# Ship to a node (NEVER reuse the `latest` tag -- docker load re-tags the
# target, which would silently swap the Qwen image out from under a node):
#   docker save officeqa-slime:gemma4 | ssh <node> 'sudo docker load'
FROM slimerl/slime:latest

# 5.13.1 is the version verified to parse gemma4_unified. sglang pins
# transformers==5.8.1 and megatron-bridge pins <=5.6.0, so pip reports both as
# conflicts; the conflict is confined to the config backfills patched below,
# and SGLang generation was verified correct against 5.13.1 (2026-08-28).
RUN pip install --no-cache-dir "transformers==5.13.1"

# sglang and megatron.bridge backfill HF configs their pinned transformers
# lacks. 5.13 ships those natively, so each backfilled register() raises
# "already used by a Transformers config". The script makes all five
# idempotent and asserts it patched them, so the build fails loudly if
# upstream moves the call sites.
COPY patch_auto_registrations.py /opt/patch_auto_registrations.py
RUN python3 /opt/patch_auto_registrations.py

# Build-time proof that the two things this image exists for actually work:
# the config type parses, and importing SGLang's Engine (the exact point the
# registration collision used to abort) succeeds.
RUN python3 -c "\
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M; \
assert 'gemma4_unified' in M, 'transformers cannot parse gemma4_unified'; \
print('OK: transformers knows gemma4_unified')" \
 && python3 -c "\
from sglang import Engine; \
import sglang, transformers; \
print(f'OK: sglang {sglang.__version__} + transformers {transformers.__version__}')"

LABEL org.opencontainers.image.base.name="slimerl/slime:latest" \
      officeqa.purpose="Gemma 4 training/rollout; transformers 5.13.1 for gemma4_unified" \
      officeqa.built="2026-08-28"
