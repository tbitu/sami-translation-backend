# Multi-arch GPU base with PyTorch preinstalled (includes Arm SBSA builds).
# Default is an NVIDIA Optimized PyTorch (NGC) container.
# See: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
                ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT: For NVIDIA Arm platforms (e.g., GB10), the CUDA-enabled PyTorch
# build is shipped with the NGC base image and is tied to the base Python.
# Using a separate Python (e.g., via micromamba) often pulls a CPU-only torch
# wheel, which makes `torch.cuda.is_available()` false even when GPUs are
# mounted. Therefore, we intentionally use the base image Python.

COPY requirements.txt /app/requirements.txt

# fairseq==0.12.2 depends on hydra/omegaconf versions whose historical wheel
# metadata is rejected by pip>=24.1. Pin pip to keep installation reliable.
RUN python -m pip install --no-cache-dir 'pip<24.1'

# IMPORTANT: Do not let pip replace the CUDA-enabled PyTorch that ships with
# the NGC base image. Some dependency resolution paths may attempt to install a
# CPU-only torch wheel from PyPI.
#
# Strategy:
# 1) Install all non-Fairseq requirements normally.
# 2) Install fairseq (and accelerate) with --no-deps, relying on the base torch.
RUN set -eux; \
    grep -vE '^(fairseq|accelerate)==|^(fairseq|accelerate)\s*$' /app/requirements.txt > /tmp/requirements.no_fairseq.txt; \
    python -m pip install --no-cache-dir -r /tmp/requirements.no_fairseq.txt; \
    python -m pip install --no-cache-dir --no-deps fairseq==0.12.2 accelerate==0.25.0; \
    # fairseq dependencies (installed explicitly so pip doesn't try to manage torch)
    python -m pip install --no-cache-dir \
        bitarray \
        portalocker \
        regex \
        sacrebleu \
        tqdm; \
    rm -f /tmp/requirements.no_fairseq.txt

# Sanity check: ensure we're still on a CUDA-enabled torch build.
RUN python - <<"PY"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("ERROR: torch.version.cuda is None (CPU-only torch installed); refusing to build")
PY

# fairseq==0.12.2 pulls in hydra-core==1.0.x which is not compatible with
# Python 3.11 (dataclass errors at import time). Upgrade to Python 3.11-compatible
# versions. This intentionally overrides fairseq's declared dependency bounds.
RUN python -m pip install --no-cache-dir -U hydra-core==1.3.2 omegaconf==2.3.0

# fairseq==0.12.2 has dataclass issues in FairseqConfig (mutable
# defaults like `common: CommonConfig = CommonConfig()`). Patch in-place to use
# default_factory so the module imports cleanly.
RUN python - <<"PY"
from __future__ import annotations

import re
from pathlib import Path

import importlib.util

spec = importlib.util.find_spec("fairseq")
if not spec or not spec.origin:
    raise RuntimeError("Unable to locate fairseq package for patching")

fairseq_root = Path(spec.origin).resolve().parent

path = fairseq_root / "dataclass" / "configs.py"
text = path.read_text(encoding="utf-8")

lines = text.splitlines(keepends=True)
out: list[str] = []

in_fairseq_config = False
indent_re = re.compile(r"^(\s{4})([A-Za-z_][A-Za-z0-9_]*):\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\(\)\s*$")

for line in lines:
    if line.startswith("class FairseqConfig"):
        in_fairseq_config = True
        out.append(line)
        continue

    if in_fairseq_config:
        # End when we hit the next top-level class/def.
        if (not line.startswith(" ")) and (line.startswith("class ") or line.startswith("def ") or line.startswith("@")):
            in_fairseq_config = False
            out.append(line)
            continue

        m = indent_re.match(line.rstrip("\n"))
        if m:
            indent, name, type_name, ctor = m.groups()
            if type_name == ctor and type_name.endswith("Config"):
                out.append(f"{indent}{name}: {type_name} = field(default_factory={type_name})\n")
                continue

    out.append(line)

patched = "".join(out)
if patched == text:
    raise RuntimeError(f"Did not patch {path}; expected FairseqConfig defaults were not found")

path.write_text(patched, encoding="utf-8")
print(f"Patched fairseq dataclass defaults in {path}")
PY

# Patch other fairseq dataclass configs that use `foo: Type = Type()` defaults
# (not allowed in Python 3.11 dataclasses). Convert them to default_factory.
RUN python - <<"PY"
from __future__ import annotations

import re
from pathlib import Path

import importlib.util

spec = importlib.util.find_spec("fairseq")
if not spec or not spec.origin:
    raise RuntimeError("Unable to locate fairseq package for patching")

root = Path(spec.origin).resolve().parent
direct_default_pattern = re.compile(
    r"^(?P<indent>\s+)(?P<name>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P=type)\(\)\s*$",
    re.MULTILINE,
)

field_default_call_pattern = re.compile(
    r"default\s*=\s*(?P<type>[A-Za-z_][A-Za-z0-9_]*)\(\)",
    re.MULTILINE,
)

patched_files = 0
patched_repls = [0]

for file_path in root.rglob("*.py"):
    text = file_path.read_text(encoding="utf-8")
    # Only patch files that already use dataclasses.field to avoid introducing
    # new imports in many modules.
    if " field" not in text and "field," not in text and "field)" not in text:
        continue

    def repl(m: re.Match[str]) -> str:
        type_name = m.group("type")
        if not (type_name.endswith("Config") or type_name.endswith("BaseConfig")):
            return m.group(0)
        patched_repls[0] += 1
        return f"{m.group('indent')}{m.group('name')}: {type_name} = field(default_factory={type_name})"

    new_text = direct_default_pattern.sub(repl, text)

    def repl_field_default(m: re.Match[str]) -> str:
        type_name = m.group("type")
        if not (type_name.endswith("Config") or type_name.endswith("BaseConfig")):
            return m.group(0)
        patched_repls[0] += 1
        return f"default_factory={type_name}"

    new_text = field_default_call_pattern.sub(repl_field_default, new_text)
    if new_text != text:
        file_path.write_text(new_text, encoding="utf-8")
        patched_files += 1

print(f"Patched {patched_repls[0]} dataclass defaults across {patched_files} fairseq files")
PY

# fairseq imports Hydra at module import time. The hydra-core version pulled in
# by fairseq (1.0.x) is not compatible with Python 3.11 and fails during
# dataclass processing. For our usage (loading TransformerModel + generating),
# Hydra initialization is not required, so make it best-effort.
RUN python - <<"PY"
from __future__ import annotations

from pathlib import Path

import importlib.util

spec = importlib.util.find_spec("fairseq")
if not spec or not spec.origin:
    raise RuntimeError("Unable to locate fairseq package for patching")

fairseq_root = Path(spec.origin).resolve().parent

init_path = fairseq_root / "__init__.py"
text = init_path.read_text(encoding="utf-8")

old = "# initialize hydra\nfrom fairseq.dataclass.initialize import hydra_init\n\nhydra_init()\n"
new = (
    "# initialize hydra\n"
    "try:\n"
    "    from fairseq.dataclass.initialize import hydra_init\n"
    "    hydra_init()\n"
    "except Exception:\n"
    "    # Hydra is optional for this service; skip if incompatible.\n"
    "    pass\n"
)

if old not in text:
    raise RuntimeError(f"Unexpected fairseq __init__.py; cannot find Hydra init block: {init_path}")

init_path.write_text(text.replace(old, new), encoding="utf-8")
print(f"Patched fairseq Hydra init in {init_path}")
PY

# fairseq assumes torch.__version__ is strictly numeric (e.g. 2.1.0). NGC often
# ships alpha/dev builds like 2.10.0a0+..., which breaks int() parsing.
# Patch to extract the numeric prefix from the patch component.
RUN python - <<"PY"
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

spec = importlib.util.find_spec("fairseq")
if not spec or not spec.origin:
    raise RuntimeError("Unable to locate fairseq package for patching")

root = Path(spec.origin).resolve().parent

def patch_file(path: Path, old: str, new: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Clean up from any previous bad patch attempt that inserted a literal "\\1".
    lines = [ln for ln in lines if ln.strip() != "\\1"]

    # Ensure we have `import re` near the top (after typing import is fine).
    has_import_re = any(ln.startswith("import re") for ln in lines)
    if not has_import_re:
        for idx, ln in enumerate(lines):
            if ln.startswith("from typing import"):
                lines.insert(idx + 1, "\nimport re\n")
                break

    text = "".join(lines)
    if old not in text:
        raise RuntimeError(f"Expected torch version parsing line not found in {path}")
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(f"Patched torch version parsing in {path}")

# transformer_layer
patch_file(
    root / "modules" / "transformer_layer.py",
    "+ int(self.torch_version[2])",
    "+ int(re.match(r\"\\d+\", self.torch_version[2]).group(0))",
)

# transformer_encoder
patch_file(
    root / "models" / "transformer" / "transformer_encoder.py",
    "+ int(torch_version[2])",
    "+ int(re.match(r\"\\d+\", torch_version[2]).group(0))",
)
PY

# Final sanity check: fairseq should import and expose TransformerModel.
RUN python - <<"PY"
import fairseq
from fairseq.models.transformer import TransformerModel
print("fairseq", fairseq.__version__, "TransformerModel_ok")
PY

COPY main.py translation_service.py /app/

# HuggingFace cache location (mount a volume here in Docker/K8s)
ENV HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_CACHE_DIR=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers

RUN mkdir -p /data/hf && chmod 0777 /data/hf

EXPOSE 8000

# Single worker to avoid duplicating large models in memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
